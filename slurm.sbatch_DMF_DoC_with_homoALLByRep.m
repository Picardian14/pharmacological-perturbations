#!/bin/bash
#SBATCH --time=360:00:00
#SBATCH --job-name=ALLtrainByRep
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-15
#SBATCH --output=outputs/ALLtrainByRep_A%A_%a.out
#SBATCH --error=outputs/ALLtrainByRep_A%A_%a.err

#Load Matlab 2017a module
ml MATLAB
ml libxdamage/1.1.4-tf2ixg3

matlab -nojvm -nodisplay<<-EOF


%struc = load("notpassedcnt.mat");
%not_passed_betas = struc.not_passed_cnt;
%idx_clust=str2num(getenv('SLURM_ARRAY_TASK_ID'));
%s = not_passed_betas(idx_clust);
REP=str2num(getenv('SLURM_ARRAY_TASK_ID'))
% Use the job ID as the seed (replace with your job identifier)

job_id = getenv('SLURM_JOB_ID'); 
org_job_id = "8498859"+REP

seed = str2double(org_job_id);

% Set the seed for the random number generator
rng(seed);

%% Read data..SC FC and time series of BOLD
%%%%%%%%%%  
load('ts_coma24_AAL_symm_withSC.mat')

C=SC;
C=C/max(max(C))*0.2;

ts_CNT=timeseries_CNT24_symm;
ts_UWS=timeseries_UWS24_symm;
ts_MCS=timeseries_MCS24_symm;

[FCempCNT,cotsamplingdataCNT,GBCempCNT] = GenerateEmpiricals(ts_CNT,C);
[FCempUWS,cotsamplingdataUWS,GBCempUWS] = GenerateEmpiricals(ts_UWS,C);
[FCempMCS,cotsamplingdataMCS,GBCempMCS] = GenerateEmpiricals(ts_MCS,C);

flp = .008;           % lowpass frequency of filter
fhi = .08;           % highpass
delt = 2.4;            % sampling interval
k=2;                  % 2nd order butterworth filter
fnq=1/(2*delt);       % Nyquist frequency
Wn=[flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn);   % construct the filter

%%%%%%%%%%%%%%%%%%
%% Parameters for the mean field model
dtt   = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
dt=0.1;

taon=100;
taog=10;
gamma=0.641;
sigma=0.01;
JN=0.15;
I0=0.382;
Jexte=1.;
Jexti=0.7;
w=1.4;
Tmax=192;
%%%%%%%%%%%%
%% Optimize
%%


WE=0:0.01:2.5;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model FC and FCD
% I put REP+20-1 to let it be clear that there were 20 iterations done before
start_point = length(dir(sprintf('./Results/G_CNT/%d/',REP+20-1)))-1 % The number of written files (including the ./ and ../ files.)  - 1 gives the last achieved value of s
for s=start_point:251 % Statisitcal repetition
    we=WE(s);
    N=90;
    Isubdiag = find(tril(ones(N),-1));
    alpha=0;
    beta=0;
    gain=1
    %Jbal=Balance_J_gain(we,C,gain);
    %save(sprintf('Results/JBAL_CNT/G%02d.mat',s), 'Jbal');
    loadedJbal = load(sprintf('Results/JBAL_CNT/G%03d.mat',s)); % It doesnt change between iterations
    Jbal = loadedJbal.Jbal;
    kk=1;
    for nsub=1:20 % Since we are doing all at the same time we
        nsub
        neuro_act=zeros(round(1000*(Tmax+60)*2.4+1),N);
        sn=0.001*ones(N,1);
        sg=0.001*ones(N,1);
        nn=1;
        for t=0:dt:(1000*(Tmax+60)*2.4)
            xn=I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg;
            xg=I0*Jexti+JN*sn-sg;
            rn=phie_gain(xn,gain);
            rg=phii_gain(xg,gain);
            sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
            sn(sn>1) = 1;
            sn(sn<0) = 0;
            sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
            sg(sg>1) = 1;
            sg(sg<0) = 0;
            j=j+1;
            if abs(mod(t,1))<0.01
                neuro_act(nn,:)=rn';
                nn=nn+1;
            end
        end
        nn=nn-1;
        
    %%%% BOLD empirical
        % Friston BALLOON MODEL
        T = nn*dtt; % Total time in seconds
        
        B = BOLD(T,neuro_act(1:nn,1)'); % B=BOLD activity, bf=Foutrier transform, f=frequency range)
        BOLD_act = zeros(length(B),N);
        BOLD_act(:,1) = B;
        
        for nnew=2:N
            B = BOLD(T,neuro_act(1:nn,nnew));
            BOLD_act(:,nnew) = B;
        end
        
        bds=BOLD_act(20:2400:end-10,:);
        FC_simul2(nsub,:,:)=corrcoef(bds);
        
        Tmax2=size(bds,1);
        Phase_BOLD_sim=zeros(N,Tmax2);
        BOLDsim=bds';
        for seed=1:N
            BOLDsim(seed,:)=BOLDsim(seed,:)-mean(BOLDsim(seed,:));
            signal_filt_sim =filtfilt(bfilt2,afilt2,BOLDsim(seed,:));
            timeserie(seed,:)=signal_filt_sim;
        end
        
        ii2=1;
        for t=1:18:Tmax2-80
            jj2=1;
            cc=corrcoef((timeserie(:,t:t+80))');
            for t2=1:18:Tmax2-80
                cc2=corrcoef((timeserie(:,t2:t2+80))');
                ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
                if jj2>ii2
                    cotsamplingsim(kk)=ca(2);  %% FCD simulation
                    kk=kk+1;
                end
                jj2=jj2+1;
            end
            ii2=ii2+1;
        end
        mFR(nsub,:) = mean(neuro_act(1:nn,:))';
    end



    [FCfittCNT,FCfittCNTSSIM, GBCfittCNT, FCDfittCNT, FC_simul]=GenerateErrors(FC_simul2,cotsamplingsim,FCempCNT,cotsamplingdataCNT,GBCempCNT);
    [FCfittMCS,FCfittMCSSSIM, GBCfittMCS, FCDfittMCS, FC_simul]=GenerateErrors(FC_simul2,cotsamplingsim,FCempMCS,cotsamplingdataMCS,GBCempMCS);
    [FCfittUWS,FCfittUWSSSIM, GBCfittUWS, FCDfittUWS, FC_simul]=GenerateErrors(FC_simul2,cotsamplingsim,FCempUWS,cotsamplingdataUWS,GBCempUWS);
    FC_simulCNT=squeeze(mean(FC_simul2(1:13, :, :),1));
    FCfittCNT_SSIM = ssim(FCempCNT, FC_simulCNT);
    FC_simulMCS=squeeze(mean(FC_simul2(1:11, :, :),1));
    FCfittMCS_SSIM = ssim(FCempMCS, FC_simulMCS);
    FC_simulUWS=squeeze(mean(FC_simul2(1:10, :, :),1));
    FCfittUWS_SSIM = ssim(FCempUWS, FC_simulUWS);
    
    %%%
    %s=1;
    %save(sprintf('CNT_model_%03d.mat',s),'GBCfitt','FCfitt','FC_simul','mFR','FC_emp');

    save(sprintf('Results/G_CNT/%02d/CNT_Gexplore_%03d.mat',REP-1+20,s),'FCDfittCNT','GBCfittCNT','FCfittCNT','FCfittCNTSSIM',"FCfittCNT_SSIM","FC_simulCNT",'FC_simul','mFR','FCempCNT');
    save(sprintf('Results/G_MCS/%02d/MCS_Gexplore_%03d.mat',REP-1+20,s),'FCDfittMCS','GBCfittMCS','FCfittMCS','FCfittMCSSSIM',"FCfittMCS_SSIM","FC_simulMCS",'FC_simul','mFR','FCempMCS');
    save(sprintf('Results/G_UWS/%02d/UWS_Gexplore_%03d.mat',REP-1+20,s),'FCDfittUWS','GBCfittUWS','FCfittUWS','FCfittUWSSSIM',"FCfittUWS_SSIM","FC_simulUWS",'FC_simul','mFR','FCempUWS');
end
EOF

