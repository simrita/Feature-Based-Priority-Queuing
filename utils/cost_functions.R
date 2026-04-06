### ml_algo is an n*k matrix, where 1 to n are true types and 1 to k are classified types. We will write a code for the case of non-preemptive priority queue, that prescribes the priority 
### for given  algo and parameters
library(lhs)
library(MASS)
true_k=function(k,n){
  true=matrix(0,nrow=n,ncol=k)
  qq=n/k
  for(i in 1:n){
    pp=ceiling(i/qq)
    true[i,pp]=1
  }
  return(true) 
}
alpha_k=function(alpha,k,n){
  true=matrix((1-alpha)/(k-1),nrow=n,ncol=k)
  qq=n/k
  for(i in 1:n){
    pp=ceiling(i/qq)
    true[i,pp]=alpha
  }
  return(true) 
}
priority_sequence=function(ml_algo,lamda,service,costs){
  n=nrow(ml_algo)
  k=ncol(ml_algo)
  priority=vector()
  for(i in 1:k){
    priority[i]=sum(costs*ml_algo[,i]*lamda)/rho_k(i,ml_algo,lamda,service)
  }
  if(length(unique(round(priority,5)))==k){
    priority<-rank(priority)
    set=vector()
    for(m in k:1){
      set[k+1-m]=which(priority==m)
    }
    return(set)
  }else{
    priority<-priority+0.0001*runif(k,0,1)
    priority<-rank(priority)
    set=vector()
    for(m in k:1){
      set[k+1-m]=which(priority==m)
    }
    return(set)
  }
}

rho_k<-function(k,ml_algo,lamda,service){
  return(sum(lamda*ml_algo[,k]/service))
}
rho_ik<-function(i,k,ml_algo,lamda,service){
  return(ml_algo[i,k]*lamda[i]/service[i])
}
mean_residual<-function(ml_algo,lamda,service){
  return(sum(lamda/(service^2)))
}
net_arrival_k<-function(k,ml_algo,lamda){
  return(sum(ml_algo[,k]*lamda))
}
mean_service_k<-function(k,ml_algo,lamda,service){
  return(sum((ml_algo[,k]*lamda)/(service))/net_arrival_k(k,ml_algo,lamda))
}
queue_k<-function(k,ml_algo,lamda,service,costs,priority){
  pos<-which(priority==k)
  if(pos==1){
    den1=1
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }else{
    den1=1-sum(unlist(lapply(priority[1:(pos-1)], function(x) rho_k(x,ml_algo,lamda,service))))
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }
  return(net_arrival_k(k,ml_algo,lamda)*mean_residual(ml_algo,lamda,service)/(den1*den2))
}
queue_ik<-function(i,k,ml_algo,lamda,service,costs,priority){
  pos<-which(priority==k)
  if(pos==1){
    den1=1
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }else{
    den1=1-sum(unlist(lapply(priority[1:(pos-1)], function(x) rho_k(x,ml_algo,lamda,service))))
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }
  return(lamda[i]*ml_algo[i,k]*mean_residual(ml_algo,lamda,service)/(den1*den2))
}

wait_k<-function(j,ml_algo,lamda,service,costs,priority){
  pos<-which(priority==j)
  if(pos==1){
    den1=1
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }else{
    den1=1-sum(unlist(lapply(priority[1:(pos-1)], function(x) rho_k(x,ml_algo,lamda,service))))
    den2=1-sum(unlist(lapply(priority[1:pos], function(x) rho_k(x,ml_algo,lamda,service))))
  }
  return(mean_residual(ml_algo,lamda,service)/(den1*den2))
}
net_cost_2<-function(ml_algo,lamda,service,costs){
  n<-nrow(ml_algo)
  k<-ncol(ml_algo)
 # priority<-priority_sequence(ml_algo,lamda,service,costs)
  #priority<-c(1:ncol(ml_algo))
  priority=c(1,2,3)
  sim<-0
  for(i in 1:n){
    for(j in 1:k){
      sim<-sim+costs[i]*lamda[i]*ml_algo[i,j]*wait_k(j,ml_algo,lamda,service,costs,priority)
    }
  }
  return(sim)
}
net_cost_priority<-function(ml_algo,priority,lamda,service,costs){
  n<-nrow(ml_algo)
  k<-ncol(ml_algo)
  sim<-0
  for(i in 1:n){
    for(j in 1:k){
      sim<-sim+costs[i]*lamda[i]*ml_algo[i,j]*wait_k(j,ml_algo,lamda,service,costs,priority)
    }
  }
  return(sim)
}
net_cost_2_wait<-function(ml_algo,lamda,service,costs){
  n<-nrow(ml_algo)
  k<-ncol(ml_algo)
  priority<-c(1,2)
  sim<-0
  for(i in 1:n){
    for(j in 1:k){
      sim<-sim+costs[i]*lamda[i]*ml_algo[i,j]*wait_k(j,ml_algo,lamda,service,costs,priority)
    }
  }
  return(sim)
}

sample_classification_matrix=function(tv_13,tv_12,tv_23){
  p1=runif(3)
  p1=p1/sum(p1)
  p21=max(0,p1[1]-tv_12)+runif(1)*(min(1,p1[1]+tv_12)-max(0,p1[1]-tv_12))
  p22=max(0,p1[2]-tv_12)+runif(1)*(min(1-p21,p1[2]+tv_12)-max(0,p1[2]-tv_12))
  p23=1-p21-p22
  p2=c(p21,p22,p23)
  p31=max(0,p1[1]-tv_13,p21-tv_23)+runif(1)*(min(1,p21+tv_23,p1[1]+tv_13)-max(0,p1[1]-tv_13,p21-tv_23))
  p32=max(0,p1[2]-tv_13,p22-tv_23)+runif(1)*(min(1-p31,p22+tv_23,p1[2]+tv_13)-max(0,p1[2]-tv_13,p22-tv_23))
  p33=1-p31-p32
  p3=c(p31,p32,p33)
  pmat=rbind(p1,p2,p3)
  return(pmat)
  }



