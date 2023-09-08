
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

	orientation(lb2gt_4,130,599). coords(lb2gt_4,147,30,599).  visible(lb2gt_4,599). happensAt(walking(lb2gt_4),599). orientation(lb2gt_6,151,599). coords(lb2gt_6,72,122,599).  visible(lb2gt_6,599). happensAt(inactive(lb2gt_6),599).
orientation(lb2gt_4,130,600). coords(lb2gt_4,148,29,600).  visible(lb2gt_4,600). happensAt(walking(lb2gt_4),600). orientation(lb2gt_6,151,600). coords(lb2gt_6,72,122,600).  visible(lb2gt_6,600). happensAt(inactive(lb2gt_6),600).
orientation(lb2gt_4,130,601). coords(lb2gt_4,148,28,601).  visible(lb2gt_4,601). happensAt(walking(lb2gt_4),601). orientation(lb2gt_6,151,601). coords(lb2gt_6,72,122,601).  visible(lb2gt_6,601). happensAt(inactive(lb2gt_6),601).
orientation(lb2gt_4,130,602). coords(lb2gt_4,148,28,602).  visible(lb2gt_4,602). happensAt(walking(lb2gt_4),602). orientation(lb2gt_6,151,602). coords(lb2gt_6,72,122,602).  visible(lb2gt_6,602). happensAt(inactive(lb2gt_6),602).
orientation(lb2gt_4,130,603). coords(lb2gt_4,149,27,603).  visible(lb2gt_4,603). happensAt(walking(lb2gt_4),603). orientation(lb2gt_6,151,603). coords(lb2gt_6,72,122,603).  visible(lb2gt_6,603). happensAt(inactive(lb2gt_6),603).
orientation(lb2gt_4,130,604). coords(lb2gt_4,149,26,604).  visible(lb2gt_4,604). happensAt(walking(lb2gt_4),604). orientation(lb2gt_6,151,604). coords(lb2gt_6,72,122,604).  visible(lb2gt_6,604). happensAt(inactive(lb2gt_6),604).
orientation(lb2gt_4,130,605). coords(lb2gt_4,150,26,605).  visible(lb2gt_4,605). happensAt(walking(lb2gt_4),605). orientation(lb2gt_6,151,605). coords(lb2gt_6,72,122,605).  visible(lb2gt_6,605). happensAt(inactive(lb2gt_6),605).
orientation(lb2gt_4,130,606). coords(lb2gt_4,151,25,606).  visible(lb2gt_4,606). happensAt(walking(lb2gt_4),606). orientation(lb2gt_6,151,606). coords(lb2gt_6,72,122,606).  visible(lb2gt_6,606). happensAt(inactive(lb2gt_6),606).
orientation(lb2gt_4,130,607). coords(lb2gt_4,151,25,607).  visible(lb2gt_4,607). happensAt(walking(lb2gt_4),607). orientation(lb2gt_6,151,607). coords(lb2gt_6,72,122,607).  visible(lb2gt_6,607). happensAt(inactive(lb2gt_6),607).
orientation(lb2gt_4,130,608). coords(lb2gt_4,151,25,608).  visible(lb2gt_4,608). happensAt(walking(lb2gt_4),608). orientation(lb2gt_6,151,608). coords(lb2gt_6,72,122,608).  visible(lb2gt_6,608). happensAt(inactive(lb2gt_6),608).
orientation(lb2gt_4,130,609). coords(lb2gt_4,151,24,609).  visible(lb2gt_4,609). happensAt(walking(lb2gt_4),609). orientation(lb2gt_6,151,609). coords(lb2gt_6,72,122,609).  visible(lb2gt_6,609). happensAt(inactive(lb2gt_6),609).
orientation(lb2gt_4,130,610). coords(lb2gt_4,152,24,610).  visible(lb2gt_4,610). happensAt(walking(lb2gt_4),610). orientation(lb2gt_6,151,610). coords(lb2gt_6,72,122,610).  visible(lb2gt_6,610). happensAt(inactive(lb2gt_6),610).
orientation(lb2gt_4,130,611). coords(lb2gt_4,152,24,611).  visible(lb2gt_4,611). happensAt(walking(lb2gt_4),611). orientation(lb2gt_6,151,611). coords(lb2gt_6,72,122,611).  visible(lb2gt_6,611). happensAt(inactive(lb2gt_6),611).
orientation(lb2gt_4,130,612). coords(lb2gt_4,153,23,612).  visible(lb2gt_4,612). happensAt(walking(lb2gt_4),612). orientation(lb2gt_6,151,612). coords(lb2gt_6,72,122,612).  visible(lb2gt_6,612). happensAt(inactive(lb2gt_6),612).
orientation(lb2gt_4,130,613). coords(lb2gt_4,153,23,613).  visible(lb2gt_4,613). happensAt(walking(lb2gt_4),613). orientation(lb2gt_6,151,613). coords(lb2gt_6,72,122,613).  visible(lb2gt_6,613). happensAt(inactive(lb2gt_6),613).
orientation(lb2gt_4,130,614). coords(lb2gt_4,154,22,614).  visible(lb2gt_4,614). happensAt(walking(lb2gt_4),614). orientation(lb2gt_6,151,614). coords(lb2gt_6,72,122,614).  visible(lb2gt_6,614). happensAt(inactive(lb2gt_6),614).
orientation(lb2gt_4,130,615). coords(lb2gt_4,155,21,615).  visible(lb2gt_4,615). happensAt(walking(lb2gt_4),615). orientation(lb2gt_6,151,615). coords(lb2gt_6,72,122,615).  visible(lb2gt_6,615). happensAt(inactive(lb2gt_6),615).
orientation(lb2gt_4,130,616). coords(lb2gt_4,155,21,616).  visible(lb2gt_4,616). happensAt(walking(lb2gt_4),616). orientation(lb2gt_6,151,616). coords(lb2gt_6,72,122,616).  visible(lb2gt_6,616). happensAt(inactive(lb2gt_6),616).
orientation(lb2gt_4,130,617). coords(lb2gt_4,155,21,617).  visible(lb2gt_4,617). happensAt(walking(lb2gt_4),617). orientation(lb2gt_6,151,617). coords(lb2gt_6,72,122,617).  visible(lb2gt_6,617). happensAt(inactive(lb2gt_6),617).
orientation(lb2gt_4,130,618). coords(lb2gt_4,155,20,618).  visible(lb2gt_4,618). happensAt(walking(lb2gt_4),618). orientation(lb2gt_6,151,618). coords(lb2gt_6,72,122,618).  visible(lb2gt_6,618). happensAt(inactive(lb2gt_6),618).
orientation(lb2gt_4,130,619). coords(lb2gt_4,157,20,619).  visible(lb2gt_4,619). happensAt(walking(lb2gt_4),619). orientation(lb2gt_6,151,619). coords(lb2gt_6,72,122,619).  visible(lb2gt_6,619). happensAt(inactive(lb2gt_6),619).
orientation(lb2gt_4,130,620). coords(lb2gt_4,158,20,620).  visible(lb2gt_4,620). happensAt(walking(lb2gt_4),620). orientation(lb2gt_6,151,620). coords(lb2gt_6,72,122,620).  visible(lb2gt_6,620). happensAt(inactive(lb2gt_6),620).
orientation(lb2gt_4,130,621). coords(lb2gt_4,158,20,621).  visible(lb2gt_4,621). happensAt(walking(lb2gt_4),621). orientation(lb2gt_6,151,621). coords(lb2gt_6,72,122,621).  visible(lb2gt_6,621). happensAt(inactive(lb2gt_6),621).
orientation(lb2gt_4,130,622). coords(lb2gt_4,159,20,622).  visible(lb2gt_4,622). happensAt(walking(lb2gt_4),622). orientation(lb2gt_6,151,622). coords(lb2gt_6,72,122,622).  visible(lb2gt_6,622). happensAt(inactive(lb2gt_6),622).
orientation(lb2gt_4,130,623). coords(lb2gt_4,160,20,623).  visible(lb2gt_4,623). happensAt(walking(lb2gt_4),623). orientation(lb2gt_6,151,623). coords(lb2gt_6,72,122,623).  visible(lb2gt_6,623). happensAt(inactive(lb2gt_6),623).
#show holdsAt/2.