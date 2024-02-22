#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=FCGenerationUWS
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-50
#SBATCH --output=outputs/FC_UWS_A%A_%a.out
#SBATCH --error=outputs/FC_UWS_A%A_%a.err

#Load Matlab 2017a module
ml MATLAB

matlab -nodisplay<<-EOF

s=str2num(getenv('SLURM_ARRAY_TASK_ID'))
job_id = getenv('SLURM_JOB_ID'); % For Slurm, get the job ID from environment
seed = str2double(job_id);
rng(seed);

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
NREP = 20;

Isubdiag = find(tril(ones(N),-1));

%%%%%%%%%%%%%%

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

%%%%%%%%%%%%

we=1.59;
G = 0:0.01:2.5;

alpha=0;
beta=0;
gain=1;%;+alpha+beta*ratio;
Jbal=Balance_J_gain(we,C,gain);
%g_index = find(abs(G - we) < 1e-6);
%loadedJbal = load(sprintf('Results/JBAL_CNT/G%03d.mat',g_index)); % It doesnt change between iterations
%Jbal = loadedJbal.Jbal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model FC and FCD


for nsub=1:NREP
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
    %save(sprintf('Results/FCGeneration/UWS_FC_generation%d.mat', s),'FC_simul2');

end



%%%
%s=1;
%save(sprintf('CNT_model_%03d.mat',s),'GBCfitt','FCfitt','FC_simul','mFR','FC_emp');

save(sprintf('Results/FCGeneration/G159_UWS/UWS_G159_%d.mat', s+100),'FC_simul2');

EOF
