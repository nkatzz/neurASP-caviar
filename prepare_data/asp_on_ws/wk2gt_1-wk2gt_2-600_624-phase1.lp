
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

	orientation(wk2gt_1,125,600). coords(wk2gt_1,115,53,600).  visible(wk2gt_1,600). happensAt(inactive(wk2gt_1),600). orientation(wk2gt_2,142,600). coords(wk2gt_2,99,63,600).  visible(wk2gt_2,600). happensAt(inactive(wk2gt_2),600).
orientation(wk2gt_1,125,601). coords(wk2gt_1,115,53,601).  visible(wk2gt_1,601). happensAt(inactive(wk2gt_1),601). orientation(wk2gt_2,142,601). coords(wk2gt_2,99,63,601).  visible(wk2gt_2,601). happensAt(inactive(wk2gt_2),601).
orientation(wk2gt_1,125,602). coords(wk2gt_1,115,53,602).  visible(wk2gt_1,602). happensAt(inactive(wk2gt_1),602). orientation(wk2gt_2,142,602). coords(wk2gt_2,99,63,602).  visible(wk2gt_2,602). happensAt(inactive(wk2gt_2),602).
orientation(wk2gt_1,125,603). coords(wk2gt_1,115,53,603).  visible(wk2gt_1,603). happensAt(inactive(wk2gt_1),603). orientation(wk2gt_2,142,603). coords(wk2gt_2,99,63,603).  visible(wk2gt_2,603). happensAt(inactive(wk2gt_2),603).
orientation(wk2gt_1,125,604). coords(wk2gt_1,115,53,604).  visible(wk2gt_1,604). happensAt(inactive(wk2gt_1),604). orientation(wk2gt_2,142,604). coords(wk2gt_2,99,63,604).  visible(wk2gt_2,604). happensAt(inactive(wk2gt_2),604).
orientation(wk2gt_1,125,605). coords(wk2gt_1,115,53,605).  visible(wk2gt_1,605). happensAt(inactive(wk2gt_1),605). orientation(wk2gt_2,142,605). coords(wk2gt_2,99,63,605).  visible(wk2gt_2,605). happensAt(inactive(wk2gt_2),605).
orientation(wk2gt_1,125,606). coords(wk2gt_1,115,53,606).  visible(wk2gt_1,606). happensAt(inactive(wk2gt_1),606). orientation(wk2gt_2,142,606). coords(wk2gt_2,99,63,606).  visible(wk2gt_2,606). happensAt(inactive(wk2gt_2),606).
orientation(wk2gt_1,125,607). coords(wk2gt_1,115,53,607).  visible(wk2gt_1,607). happensAt(inactive(wk2gt_1),607). orientation(wk2gt_2,142,607). coords(wk2gt_2,99,63,607).  visible(wk2gt_2,607). happensAt(inactive(wk2gt_2),607).
orientation(wk2gt_1,125,608). coords(wk2gt_1,115,53,608).  visible(wk2gt_1,608). happensAt(inactive(wk2gt_1),608). orientation(wk2gt_2,142,608). coords(wk2gt_2,99,63,608).  visible(wk2gt_2,608). happensAt(inactive(wk2gt_2),608).
orientation(wk2gt_1,125,609). coords(wk2gt_1,115,53,609).  visible(wk2gt_1,609). happensAt(inactive(wk2gt_1),609). orientation(wk2gt_2,142,609). coords(wk2gt_2,99,63,609).  visible(wk2gt_2,609). happensAt(inactive(wk2gt_2),609).
orientation(wk2gt_1,125,610). coords(wk2gt_1,114,53,610).  visible(wk2gt_1,610). happensAt(inactive(wk2gt_1),610). orientation(wk2gt_2,142,610). coords(wk2gt_2,99,63,610).  visible(wk2gt_2,610). happensAt(inactive(wk2gt_2),610).
orientation(wk2gt_1,125,611). coords(wk2gt_1,114,53,611).  visible(wk2gt_1,611). happensAt(inactive(wk2gt_1),611). orientation(wk2gt_2,142,611). coords(wk2gt_2,99,63,611).  visible(wk2gt_2,611). happensAt(inactive(wk2gt_2),611).
orientation(wk2gt_1,125,612). coords(wk2gt_1,114,53,612).  visible(wk2gt_1,612). happensAt(inactive(wk2gt_1),612). orientation(wk2gt_2,142,612). coords(wk2gt_2,99,63,612).  visible(wk2gt_2,612). happensAt(inactive(wk2gt_2),612).
orientation(wk2gt_1,125,613). coords(wk2gt_1,114,53,613).  visible(wk2gt_1,613). happensAt(inactive(wk2gt_1),613). orientation(wk2gt_2,142,613). coords(wk2gt_2,99,63,613).  visible(wk2gt_2,613). happensAt(inactive(wk2gt_2),613).
orientation(wk2gt_1,125,614). coords(wk2gt_1,114,53,614).  visible(wk2gt_1,614). happensAt(inactive(wk2gt_1),614). orientation(wk2gt_2,142,614). coords(wk2gt_2,99,63,614).  visible(wk2gt_2,614). happensAt(inactive(wk2gt_2),614).
orientation(wk2gt_1,125,615). coords(wk2gt_1,114,53,615).  visible(wk2gt_1,615). happensAt(inactive(wk2gt_1),615). orientation(wk2gt_2,142,615). coords(wk2gt_2,99,63,615).  visible(wk2gt_2,615). happensAt(inactive(wk2gt_2),615).
orientation(wk2gt_1,125,616). coords(wk2gt_1,114,53,616).  visible(wk2gt_1,616). happensAt(inactive(wk2gt_1),616). orientation(wk2gt_2,142,616). coords(wk2gt_2,99,63,616).  visible(wk2gt_2,616). happensAt(inactive(wk2gt_2),616).
orientation(wk2gt_1,125,617). coords(wk2gt_1,115,53,617).  visible(wk2gt_1,617). happensAt(inactive(wk2gt_1),617). orientation(wk2gt_2,142,617). coords(wk2gt_2,99,63,617).  visible(wk2gt_2,617). happensAt(inactive(wk2gt_2),617).
orientation(wk2gt_1,125,618). coords(wk2gt_1,115,53,618).  visible(wk2gt_1,618). happensAt(inactive(wk2gt_1),618). orientation(wk2gt_2,142,618). coords(wk2gt_2,99,63,618).  visible(wk2gt_2,618). happensAt(inactive(wk2gt_2),618).
orientation(wk2gt_1,125,619). coords(wk2gt_1,115,53,619).  visible(wk2gt_1,619). happensAt(inactive(wk2gt_1),619). orientation(wk2gt_2,142,619). coords(wk2gt_2,99,63,619).  visible(wk2gt_2,619). happensAt(inactive(wk2gt_2),619).
orientation(wk2gt_1,125,620). coords(wk2gt_1,115,53,620).  visible(wk2gt_1,620). happensAt(inactive(wk2gt_1),620). orientation(wk2gt_2,142,620). coords(wk2gt_2,99,63,620).  visible(wk2gt_2,620). happensAt(inactive(wk2gt_2),620).
orientation(wk2gt_1,125,621). coords(wk2gt_1,115,53,621).  visible(wk2gt_1,621). happensAt(inactive(wk2gt_1),621). orientation(wk2gt_2,142,621). coords(wk2gt_2,99,63,621).  visible(wk2gt_2,621). happensAt(inactive(wk2gt_2),621).
orientation(wk2gt_1,125,622). coords(wk2gt_1,115,53,622).  visible(wk2gt_1,622). happensAt(inactive(wk2gt_1),622). orientation(wk2gt_2,142,622). coords(wk2gt_2,99,63,622).  visible(wk2gt_2,622). happensAt(inactive(wk2gt_2),622).
orientation(wk2gt_1,125,623). coords(wk2gt_1,115,53,623).  visible(wk2gt_1,623). happensAt(inactive(wk2gt_1),623). orientation(wk2gt_2,142,623). coords(wk2gt_2,99,63,623).  visible(wk2gt_2,623). happensAt(inactive(wk2gt_2),623).
orientation(wk2gt_1,125,624). coords(wk2gt_1,115,53,624).  visible(wk2gt_1,624). happensAt(inactive(wk2gt_1),624). orientation(wk2gt_2,142,624). coords(wk2gt_2,99,63,624).  visible(wk2gt_2,624). happensAt(inactive(wk2gt_2),624).
#show holdsAt/2.