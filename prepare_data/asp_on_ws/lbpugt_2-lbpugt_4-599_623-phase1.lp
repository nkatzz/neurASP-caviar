
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

	orientation(lbpugt_2,132,599). coords(lbpugt_2,181,50,599).  visible(lbpugt_2,599). happensAt(inactive(lbpugt_2),599). orientation(lbpugt_4,135,599). coords(lbpugt_4,219,34,599).  visible(lbpugt_4,599). happensAt(inactive(lbpugt_4),599).
orientation(lbpugt_2,132,600). coords(lbpugt_2,181,50,600).  visible(lbpugt_2,600). happensAt(inactive(lbpugt_2),600). orientation(lbpugt_4,135,600). coords(lbpugt_4,219,34,600).  visible(lbpugt_4,600). happensAt(inactive(lbpugt_4),600).
orientation(lbpugt_2,132,601). coords(lbpugt_2,181,50,601).  visible(lbpugt_2,601). happensAt(inactive(lbpugt_2),601). orientation(lbpugt_4,135,601). coords(lbpugt_4,219,34,601).  visible(lbpugt_4,601). happensAt(inactive(lbpugt_4),601).
orientation(lbpugt_2,132,602). coords(lbpugt_2,181,50,602).  visible(lbpugt_2,602). happensAt(inactive(lbpugt_2),602). orientation(lbpugt_4,135,602). coords(lbpugt_4,219,34,602).  visible(lbpugt_4,602). happensAt(inactive(lbpugt_4),602).
orientation(lbpugt_2,132,603). coords(lbpugt_2,181,50,603).  visible(lbpugt_2,603). happensAt(inactive(lbpugt_2),603). orientation(lbpugt_4,135,603). coords(lbpugt_4,219,34,603).  visible(lbpugt_4,603). happensAt(inactive(lbpugt_4),603).
orientation(lbpugt_2,132,604). coords(lbpugt_2,181,50,604).  visible(lbpugt_2,604). happensAt(inactive(lbpugt_2),604). orientation(lbpugt_4,135,604). coords(lbpugt_4,219,34,604).  visible(lbpugt_4,604). happensAt(inactive(lbpugt_4),604).
orientation(lbpugt_2,132,605). coords(lbpugt_2,181,50,605).  visible(lbpugt_2,605). happensAt(inactive(lbpugt_2),605). orientation(lbpugt_4,135,605). coords(lbpugt_4,219,34,605).  visible(lbpugt_4,605). happensAt(inactive(lbpugt_4),605).
orientation(lbpugt_2,132,606). coords(lbpugt_2,181,50,606).  visible(lbpugt_2,606). happensAt(inactive(lbpugt_2),606). orientation(lbpugt_4,135,606). coords(lbpugt_4,219,34,606).  visible(lbpugt_4,606). happensAt(inactive(lbpugt_4),606).
orientation(lbpugt_2,132,607). coords(lbpugt_2,181,50,607).  visible(lbpugt_2,607). happensAt(inactive(lbpugt_2),607). orientation(lbpugt_4,135,607). coords(lbpugt_4,219,34,607).  visible(lbpugt_4,607). happensAt(inactive(lbpugt_4),607).
orientation(lbpugt_2,132,608). coords(lbpugt_2,181,50,608).  visible(lbpugt_2,608). happensAt(inactive(lbpugt_2),608). orientation(lbpugt_4,135,608). coords(lbpugt_4,219,34,608).  visible(lbpugt_4,608). happensAt(inactive(lbpugt_4),608).
orientation(lbpugt_2,132,609). coords(lbpugt_2,181,50,609).  visible(lbpugt_2,609). happensAt(inactive(lbpugt_2),609). orientation(lbpugt_4,135,609). coords(lbpugt_4,219,34,609).  visible(lbpugt_4,609). happensAt(inactive(lbpugt_4),609).
orientation(lbpugt_2,132,610). coords(lbpugt_2,181,50,610).  visible(lbpugt_2,610). happensAt(inactive(lbpugt_2),610). orientation(lbpugt_4,135,610). coords(lbpugt_4,219,34,610).  visible(lbpugt_4,610). happensAt(inactive(lbpugt_4),610).
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
#show holdsAt/2.