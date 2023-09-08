
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbpugt_2,132,611). coords(lbpugt_2,181,50,611).  visible(lbpugt_2,611). happensAt(inactive(lbpugt_2),611). orientation(lbpugt_4,135,611). coords(lbpugt_4,219,34,611).  visible(lbpugt_4,611). happensAt(inactive(lbpugt_4),611).
orientation(lbpugt_2,132,612). coords(lbpugt_2,181,50,612).  visible(lbpugt_2,612). happensAt(inactive(lbpugt_2),612). orientation(lbpugt_4,135,612). coords(lbpugt_4,219,34,612).  visible(lbpugt_4,612). happensAt(inactive(lbpugt_4),612).
orientation(lbpugt_2,132,613). coords(lbpugt_2,181,50,613).  visible(lbpugt_2,613). happensAt(inactive(lbpugt_2),613). orientation(lbpugt_4,135,613). coords(lbpugt_4,219,34,613).  visible(lbpugt_4,613). happensAt(inactive(lbpugt_4),613).
orientation(lbpugt_2,132,614). coords(lbpugt_2,181,50,614).  visible(lbpugt_2,614). happensAt(inactive(lbpugt_2),614). orientation(lbpugt_4,135,614). coords(lbpugt_4,219,34,614).  visible(lbpugt_4,614). happensAt(inactive(lbpugt_4),614).
orientation(lbpugt_2,132,615). coords(lbpugt_2,181,50,615).  visible(lbpugt_2,615). happensAt(inactive(lbpugt_2),615). orientation(lbpugt_4,135,615). coords(lbpugt_4,219,34,615).  visible(lbpugt_4,615). happensAt(inactive(lbpugt_4),615).
orientation(lbpugt_2,132,616). coords(lbpugt_2,181,50,616).  visible(lbpugt_2,616). happensAt(inactive(lbpugt_2),616). orientation(lbpugt_4,135,616). coords(lbpugt_4,219,34,616).  visible(lbpugt_4,616). happensAt(inactive(lbpugt_4),616).
orientation(lbpugt_2,132,617). coords(lbpugt_2,181,50,617).  visible(lbpugt_2,617). happensAt(inactive(lbpugt_2),617). orientation(lbpugt_4,135,617). coords(lbpugt_4,219,34,617).  visible(lbpugt_4,617). happensAt(inactive(lbpugt_4),617).
orientation(lbpugt_2,132,618). coords(lbpugt_2,181,50,618).  visible(lbpugt_2,618). happensAt(inactive(lbpugt_2),618). orientation(lbpugt_4,135,618). coords(lbpugt_4,219,34,618).  visible(lbpugt_4,618). happensAt(inactive(lbpugt_4),618).
orientation(lbpugt_2,132,619). coords(lbpugt_2,181,50,619).  visible(lbpugt_2,619). happensAt(active(lbpugt_2),619). orientation(lbpugt_4,135,619). coords(lbpugt_4,219,34,619).  visible(lbpugt_4,619). happensAt(inactive(lbpugt_4),619).
orientation(lbpugt_2,132,620). coords(lbpugt_2,181,50,620).  visible(lbpugt_2,620). happensAt(active(lbpugt_2),620). orientation(lbpugt_4,135,620). coords(lbpugt_4,219,34,620).  visible(lbpugt_4,620). happensAt(inactive(lbpugt_4),620).
orientation(lbpugt_2,132,621). coords(lbpugt_2,181,50,621).  visible(lbpugt_2,621). happensAt(active(lbpugt_2),621). orientation(lbpugt_4,135,621). coords(lbpugt_4,219,34,621).  visible(lbpugt_4,621). happensAt(inactive(lbpugt_4),621).
orientation(lbpugt_2,132,622). coords(lbpugt_2,181,50,622).  visible(lbpugt_2,622). happensAt(active(lbpugt_2),622). orientation(lbpugt_4,135,622). coords(lbpugt_4,219,34,622).  visible(lbpugt_4,622). happensAt(inactive(lbpugt_4),622).
orientation(lbpugt_2,132,623). coords(lbpugt_2,181,50,623).  visible(lbpugt_2,623). happensAt(active(lbpugt_2),623). orientation(lbpugt_4,135,623). coords(lbpugt_4,219,34,623).  visible(lbpugt_4,623). happensAt(inactive(lbpugt_4),623).
orientation(lbpugt_2,132,624). coords(lbpugt_2,181,50,624).  visible(lbpugt_2,624). happensAt(active(lbpugt_2),624). orientation(lbpugt_4,135,624). coords(lbpugt_4,219,34,624).  visible(lbpugt_4,624). happensAt(inactive(lbpugt_4),624).
orientation(lbpugt_2,132,625). coords(lbpugt_2,181,50,625).  visible(lbpugt_2,625). happensAt(active(lbpugt_2),625). orientation(lbpugt_4,135,625). coords(lbpugt_4,219,34,625).  visible(lbpugt_4,625). happensAt(inactive(lbpugt_4),625).
orientation(lbpugt_2,132,626). coords(lbpugt_2,181,50,626).  visible(lbpugt_2,626). happensAt(active(lbpugt_2),626). orientation(lbpugt_4,135,626). coords(lbpugt_4,219,34,626).  visible(lbpugt_4,626). happensAt(inactive(lbpugt_4),626).
orientation(lbpugt_2,132,627). coords(lbpugt_2,181,50,627).  visible(lbpugt_2,627). happensAt(active(lbpugt_2),627). orientation(lbpugt_4,135,627). coords(lbpugt_4,219,34,627).  visible(lbpugt_4,627). happensAt(inactive(lbpugt_4),627).
orientation(lbpugt_2,132,628). coords(lbpugt_2,181,50,628).  visible(lbpugt_2,628). happensAt(active(lbpugt_2),628). orientation(lbpugt_4,135,628). coords(lbpugt_4,219,34,628).  visible(lbpugt_4,628). happensAt(inactive(lbpugt_4),628).
orientation(lbpugt_2,132,629). coords(lbpugt_2,181,50,629).  visible(lbpugt_2,629). happensAt(active(lbpugt_2),629). orientation(lbpugt_4,135,629). coords(lbpugt_4,219,34,629).  visible(lbpugt_4,629). happensAt(inactive(lbpugt_4),629).
orientation(lbpugt_2,132,630). coords(lbpugt_2,181,50,630).  visible(lbpugt_2,630). happensAt(active(lbpugt_2),630). orientation(lbpugt_4,135,630). coords(lbpugt_4,219,34,630).  visible(lbpugt_4,630). happensAt(inactive(lbpugt_4),630).
orientation(lbpugt_2,132,631). coords(lbpugt_2,181,50,631).  visible(lbpugt_2,631). happensAt(active(lbpugt_2),631). orientation(lbpugt_4,135,631). coords(lbpugt_4,219,34,631).  visible(lbpugt_4,631). happensAt(inactive(lbpugt_4),631).
orientation(lbpugt_2,132,632). coords(lbpugt_2,181,50,632).  visible(lbpugt_2,632). happensAt(active(lbpugt_2),632). orientation(lbpugt_4,135,632). coords(lbpugt_4,219,34,632).  visible(lbpugt_4,632). happensAt(inactive(lbpugt_4),632).
orientation(lbpugt_2,132,633). coords(lbpugt_2,182,51,633).  visible(lbpugt_2,633). happensAt(active(lbpugt_2),633). orientation(lbpugt_4,135,633). coords(lbpugt_4,219,34,633).  visible(lbpugt_4,633). happensAt(inactive(lbpugt_4),633).
orientation(lbpugt_2,132,634). coords(lbpugt_2,182,52,634).  visible(lbpugt_2,634). happensAt(active(lbpugt_2),634). orientation(lbpugt_4,135,634). coords(lbpugt_4,219,34,634).  visible(lbpugt_4,634). happensAt(inactive(lbpugt_4),634).
orientation(lbpugt_2,132,635). coords(lbpugt_2,182,52,635).  visible(lbpugt_2,635). happensAt(active(lbpugt_2),635). orientation(lbpugt_4,135,635). coords(lbpugt_4,219,34,635).  visible(lbpugt_4,635). happensAt(inactive(lbpugt_4),635).
#show holdsAt/2.