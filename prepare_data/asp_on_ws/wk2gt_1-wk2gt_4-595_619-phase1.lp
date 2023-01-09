
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

	orientation(wk2gt_1,125,595). coords(wk2gt_1,115,53,595).  visible(wk2gt_1,595). happensAt(inactive(wk2gt_1),595). orientation(wk2gt_4,152,595). coords(wk2gt_4,35,111,595).  visible(wk2gt_4,595). happensAt(walking(wk2gt_4),595).
orientation(wk2gt_1,125,596). coords(wk2gt_1,115,53,596).  visible(wk2gt_1,596). happensAt(inactive(wk2gt_1),596). orientation(wk2gt_4,152,596). coords(wk2gt_4,35,111,596).  visible(wk2gt_4,596). happensAt(walking(wk2gt_4),596).
orientation(wk2gt_1,125,597). coords(wk2gt_1,115,53,597).  visible(wk2gt_1,597). happensAt(inactive(wk2gt_1),597). orientation(wk2gt_4,152,597). coords(wk2gt_4,35,112,597).  visible(wk2gt_4,597). happensAt(walking(wk2gt_4),597).
orientation(wk2gt_1,125,598). coords(wk2gt_1,115,53,598).  visible(wk2gt_1,598). happensAt(inactive(wk2gt_1),598). orientation(wk2gt_4,152,598). coords(wk2gt_4,35,112,598).  visible(wk2gt_4,598). happensAt(walking(wk2gt_4),598).
orientation(wk2gt_1,125,599). coords(wk2gt_1,115,53,599).  visible(wk2gt_1,599). happensAt(inactive(wk2gt_1),599). orientation(wk2gt_4,152,599). coords(wk2gt_4,35,113,599).  visible(wk2gt_4,599). happensAt(walking(wk2gt_4),599).
orientation(wk2gt_1,125,600). coords(wk2gt_1,115,53,600).  visible(wk2gt_1,600). happensAt(inactive(wk2gt_1),600). orientation(wk2gt_4,152,600). coords(wk2gt_4,35,113,600).  visible(wk2gt_4,600). happensAt(walking(wk2gt_4),600).
orientation(wk2gt_1,125,601). coords(wk2gt_1,115,53,601).  visible(wk2gt_1,601). happensAt(inactive(wk2gt_1),601). orientation(wk2gt_4,157,601). coords(wk2gt_4,35,113,601).  visible(wk2gt_4,601). happensAt(walking(wk2gt_4),601).
orientation(wk2gt_1,125,602). coords(wk2gt_1,115,53,602).  visible(wk2gt_1,602). happensAt(inactive(wk2gt_1),602). orientation(wk2gt_4,157,602). coords(wk2gt_4,35,113,602).  visible(wk2gt_4,602). happensAt(walking(wk2gt_4),602).
orientation(wk2gt_1,125,603). coords(wk2gt_1,115,53,603).  visible(wk2gt_1,603). happensAt(inactive(wk2gt_1),603). orientation(wk2gt_4,161,603). coords(wk2gt_4,32,113,603).  visible(wk2gt_4,603). happensAt(walking(wk2gt_4),603).
orientation(wk2gt_1,125,604). coords(wk2gt_1,115,53,604).  visible(wk2gt_1,604). happensAt(inactive(wk2gt_1),604). orientation(wk2gt_4,161,604). coords(wk2gt_4,32,113,604).  visible(wk2gt_4,604). happensAt(walking(wk2gt_4),604).
orientation(wk2gt_1,125,605). coords(wk2gt_1,115,53,605).  visible(wk2gt_1,605). happensAt(inactive(wk2gt_1),605). orientation(wk2gt_4,161,605). coords(wk2gt_4,32,115,605).  visible(wk2gt_4,605). happensAt(walking(wk2gt_4),605).
orientation(wk2gt_1,125,606). coords(wk2gt_1,115,53,606).  visible(wk2gt_1,606). happensAt(inactive(wk2gt_1),606). orientation(wk2gt_4,161,606). coords(wk2gt_4,32,115,606).  visible(wk2gt_4,606). happensAt(walking(wk2gt_4),606).
orientation(wk2gt_1,125,607). coords(wk2gt_1,115,53,607).  visible(wk2gt_1,607). happensAt(inactive(wk2gt_1),607). orientation(wk2gt_4,161,607). coords(wk2gt_4,32,116,607).  visible(wk2gt_4,607). happensAt(walking(wk2gt_4),607).
orientation(wk2gt_1,125,608). coords(wk2gt_1,115,53,608).  visible(wk2gt_1,608). happensAt(inactive(wk2gt_1),608). orientation(wk2gt_4,161,608). coords(wk2gt_4,31,116,608).  visible(wk2gt_4,608). happensAt(walking(wk2gt_4),608).
orientation(wk2gt_1,125,609). coords(wk2gt_1,115,53,609).  visible(wk2gt_1,609). happensAt(inactive(wk2gt_1),609). orientation(wk2gt_4,161,609). coords(wk2gt_4,31,117,609).  visible(wk2gt_4,609). happensAt(walking(wk2gt_4),609).
orientation(wk2gt_1,125,610). coords(wk2gt_1,114,53,610).  visible(wk2gt_1,610). happensAt(inactive(wk2gt_1),610). orientation(wk2gt_4,161,610). coords(wk2gt_4,31,117,610).  visible(wk2gt_4,610). happensAt(walking(wk2gt_4),610).
orientation(wk2gt_1,125,611). coords(wk2gt_1,114,53,611).  visible(wk2gt_1,611). happensAt(inactive(wk2gt_1),611). orientation(wk2gt_4,161,611). coords(wk2gt_4,31,117,611).  visible(wk2gt_4,611). happensAt(walking(wk2gt_4),611).
orientation(wk2gt_1,125,612). coords(wk2gt_1,114,53,612).  visible(wk2gt_1,612). happensAt(inactive(wk2gt_1),612). orientation(wk2gt_4,161,612). coords(wk2gt_4,31,118,612).  visible(wk2gt_4,612). happensAt(walking(wk2gt_4),612).
orientation(wk2gt_1,125,613). coords(wk2gt_1,114,53,613).  visible(wk2gt_1,613). happensAt(inactive(wk2gt_1),613). orientation(wk2gt_4,161,613). coords(wk2gt_4,31,118,613).  visible(wk2gt_4,613). happensAt(walking(wk2gt_4),613).
orientation(wk2gt_1,125,614). coords(wk2gt_1,114,53,614).  visible(wk2gt_1,614). happensAt(inactive(wk2gt_1),614). orientation(wk2gt_4,161,614). coords(wk2gt_4,31,119,614).  visible(wk2gt_4,614). happensAt(walking(wk2gt_4),614).
orientation(wk2gt_1,125,615). coords(wk2gt_1,114,53,615).  visible(wk2gt_1,615). happensAt(inactive(wk2gt_1),615). orientation(wk2gt_4,161,615). coords(wk2gt_4,31,119,615).  visible(wk2gt_4,615). happensAt(walking(wk2gt_4),615).
orientation(wk2gt_1,125,616). coords(wk2gt_1,114,53,616).  visible(wk2gt_1,616). happensAt(inactive(wk2gt_1),616). orientation(wk2gt_4,161,616). coords(wk2gt_4,31,119,616).  visible(wk2gt_4,616). happensAt(walking(wk2gt_4),616).
orientation(wk2gt_1,125,617). coords(wk2gt_1,115,53,617).  visible(wk2gt_1,617). happensAt(inactive(wk2gt_1),617). orientation(wk2gt_4,161,617). coords(wk2gt_4,31,120,617).  visible(wk2gt_4,617). happensAt(walking(wk2gt_4),617).
orientation(wk2gt_1,125,618). coords(wk2gt_1,115,53,618).  visible(wk2gt_1,618). happensAt(inactive(wk2gt_1),618). orientation(wk2gt_4,161,618). coords(wk2gt_4,31,120,618).  visible(wk2gt_4,618). happensAt(walking(wk2gt_4),618).
orientation(wk2gt_1,125,619). coords(wk2gt_1,115,53,619).  visible(wk2gt_1,619). happensAt(inactive(wk2gt_1),619). orientation(wk2gt_4,161,619). coords(wk2gt_4,30,120,619).  visible(wk2gt_4,619). happensAt(walking(wk2gt_4),619).
#show holdsAt/2.