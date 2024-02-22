#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=NegPerturb5HT2A
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-101
#SBATCH --output=outputs/UWS_Neg_Wgain_5HT2A%A_%a.out
#SBATCH --error=outputs/UWS_Neg_Wgain_5HT2A%A_%a.err

#Load Matlab 2017a module
ml MATLAB

matlab -nojvm -nodisplay<<-EOF

%struc = load("NotPassedBetas5HTuws.mat");
%not_passed_betas = struc.not_passed_betas;
%idx_clust=str2num(getenv('SLURM_ARRAY_TASK_ID'));
%s = not_passed_betas(1, idx_clust);

s=str2num(getenv('SLURM_ARRAY_TASK_ID'))
gen=load('5HT2a_cimbi_hc29_beliveau.csv');
gen=gen(1:90);
gensym(1:45)=gen(1:2:90);
gensym(90:-1:46)=gen(2:2:90);
ratio=gensym';
ratio=ratio/(max(ratio)-min(ratio));
ratio=ratio-max(ratio)+1;
ratio(find(ratio<0))=0;

%% Read data..SC FC and time series of BOLD
%%%%%%%%%%  
load('ts_coma24_AAL_symm_withSC.mat')

C=SC;
C=C/max(max(C))*0.2;

ts=timeseries_CNT24_symm;

N=90;
NSUB=size(ts,2);
NSUBSIM=NSUB; 
Tmax=192;
indexsub=1:NSUB;

for nsub=indexsub
    tsdata(:,:,nsub)=ts{:,nsub}(:,1:Tmax)';
    FCdata(nsub,:,:)=corrcoef(squeeze(tsdata(:,:,nsub)));
end

FC_emp=squeeze(mean(FCdata,1));

FCemp2=FC_emp-FC_emp.*eye(N);
GBCemp=mean(FCemp2,2);

Isubdiag = find(tril(ones(N),-1));

%%%%%%%%%%%%%%

flp = .008;           % lowpass frequency of filter
fhi = .08;           % highpass
delt = 2.4;            % sampling interval
k=2;                  % 2nd order butterworth filter
fnq=1/(2*delt);       % Nyquist frequency
Wn=[flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn);   % construct the filter


%%%%%%%%%%%%%%
kk=1;
for nsub=1:NSUB
    BOLDdata=(squeeze(tsdata(:,:,nsub)))';
    for seed=1:N
        BOLDdata(seed,:)=BOLDdata(seed,:)-mean(BOLDdata(seed,:));
        timeseriedata(seed,:) =filtfilt(bfilt2,afilt2,BOLDdata(seed,:));
    end
 

    ii2=1;
    for t=1:18:Tmax-80
        jj2=1;
        cc=corrcoef((timeseriedata(:,t:t+80))');
        for t2=1:18:Tmax-80
            cc2=corrcoef((timeseriedata(:,t2:t2+80))');
            ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
            if jj2>ii2
                cotsamplingdata(kk)=ca(2);   %% this accumulate all elements of the FCD empirical
                kk=kk+1;
            end
            jj2=jj2+1;
        end
        ii2=ii2+1;
    end
end


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

%%%%%%%%%%%%
%% Optimize
%%


BETA=-1:0.01:0;
beta=BETA(s);


we=1.58;
alpha = 0
gain=1+alpha+beta*ratio;
if s==1
    disp("Loading default Jbal")
    G = 0:0.01:2.5;
    g_index = find(abs(G - we) < 1e-6);
    JbalLoaded = load(sprintf('Results/JBAL_CNT/G%03d.mat',g_index)); % It doesnt change between iterations
    Jbal = JbalLoaded.Jbal
else
    Jbal=Balance_J_gain(we,C,gain);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model FC and FCD
for REP=1:20
    kk=1;
    for nsub=1:NSUBSIM
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

    FC_simul=squeeze(mean(FC_simul2,1));
    cc=corrcoef(atanh(FC_emp(Isubdiag)),atanh(FC_simul(Isubdiag)));
    FCfitt=cc(2); %% FC fitting

    FCsim2=FC_simul-FC_simul.*eye(N);
    GBCsim=mean(FCsim2,2);
    GBCfitt=corr2(GBCemp,GBCsim);

    [hh pp FCDfitt]=kstest2(cotsamplingdata,cotsamplingsim);  %% FCD fitting
    %%%
    mkdir(sprintf('Results/UWS_Neg_Wgain_5HT2A/%02d',REP-1))
    save(sprintf('Results/UWS_Neg_Wgain_5HT2A/%02d/job_%03d.mat',REP-1,s),'FCDfitt','GBCfitt','FCfitt','FC_simul','mFR','FC_emp');
    %save(sprintf('Results/MCS_Wgain_5HT/job_%03d.mat',s),'GBCfitt','FCfitt','FCDfitt','FC_simul','mFR');
end
EOF

