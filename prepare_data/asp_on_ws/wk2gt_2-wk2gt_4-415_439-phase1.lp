
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

	orientation(wk2gt_2,141,415). coords(wk2gt_2,98,63,415).  visible(wk2gt_2,415). happensAt(inactive(wk2gt_2),415). orientation(wk2gt_4,149,415). coords(wk2gt_4,118,106,415).  visible(wk2gt_4,415). happensAt(walking(wk2gt_4),415).
orientation(wk2gt_2,141,416). coords(wk2gt_2,98,63,416).  visible(wk2gt_2,416). happensAt(inactive(wk2gt_2),416). orientation(wk2gt_4,149,416). coords(wk2gt_4,116,106,416).  visible(wk2gt_4,416). happensAt(walking(wk2gt_4),416).
orientation(wk2gt_2,141,417). coords(wk2gt_2,98,63,417).  visible(wk2gt_2,417). happensAt(inactive(wk2gt_2),417). orientation(wk2gt_4,149,417). coords(wk2gt_4,114,106,417).  visible(wk2gt_4,417). happensAt(walking(wk2gt_4),417).
orientation(wk2gt_2,141,418). coords(wk2gt_2,98,63,418).  visible(wk2gt_2,418). happensAt(inactive(wk2gt_2),418). orientation(wk2gt_4,149,418). coords(wk2gt_4,113,106,418).  visible(wk2gt_4,418). happensAt(walking(wk2gt_4),418).
orientation(wk2gt_2,141,419). coords(wk2gt_2,98,63,419).  visible(wk2gt_2,419). happensAt(inactive(wk2gt_2),419). orientation(wk2gt_4,149,419). coords(wk2gt_4,113,106,419).  visible(wk2gt_4,419). happensAt(walking(wk2gt_4),419).
orientation(wk2gt_2,141,420). coords(wk2gt_2,98,63,420).  visible(wk2gt_2,420). happensAt(inactive(wk2gt_2),420). orientation(wk2gt_4,149,420). coords(wk2gt_4,113,105,420).  visible(wk2gt_4,420). happensAt(walking(wk2gt_4),420).
orientation(wk2gt_2,141,421). coords(wk2gt_2,98,63,421).  visible(wk2gt_2,421). happensAt(inactive(wk2gt_2),421). orientation(wk2gt_4,149,421). coords(wk2gt_4,113,105,421).  visible(wk2gt_4,421). happensAt(walking(wk2gt_4),421).
orientation(wk2gt_2,141,422). coords(wk2gt_2,98,63,422).  visible(wk2gt_2,422). happensAt(inactive(wk2gt_2),422). orientation(wk2gt_4,149,422). coords(wk2gt_4,113,105,422).  visible(wk2gt_4,422). happensAt(walking(wk2gt_4),422).
orientation(wk2gt_2,141,423). coords(wk2gt_2,98,63,423).  visible(wk2gt_2,423). happensAt(inactive(wk2gt_2),423). orientation(wk2gt_4,149,423). coords(wk2gt_4,112,105,423).  visible(wk2gt_4,423). happensAt(walking(wk2gt_4),423).
orientation(wk2gt_2,141,424). coords(wk2gt_2,98,63,424).  visible(wk2gt_2,424). happensAt(inactive(wk2gt_2),424). orientation(wk2gt_4,149,424). coords(wk2gt_4,111,105,424).  visible(wk2gt_4,424). happensAt(walking(wk2gt_4),424).
orientation(wk2gt_2,141,425). coords(wk2gt_2,98,63,425).  visible(wk2gt_2,425). happensAt(inactive(wk2gt_2),425). orientation(wk2gt_4,149,425). coords(wk2gt_4,110,104,425).  visible(wk2gt_4,425). happensAt(walking(wk2gt_4),425).
orientation(wk2gt_2,141,426). coords(wk2gt_2,98,63,426).  visible(wk2gt_2,426). happensAt(inactive(wk2gt_2),426). orientation(wk2gt_4,149,426). coords(wk2gt_4,109,104,426).  visible(wk2gt_4,426). happensAt(walking(wk2gt_4),426).
orientation(wk2gt_2,141,427). coords(wk2gt_2,98,63,427).  visible(wk2gt_2,427). happensAt(inactive(wk2gt_2),427). orientation(wk2gt_4,149,427). coords(wk2gt_4,107,103,427).  visible(wk2gt_4,427). happensAt(walking(wk2gt_4),427).
orientation(wk2gt_2,141,428). coords(wk2gt_2,98,63,428).  visible(wk2gt_2,428). happensAt(inactive(wk2gt_2),428). orientation(wk2gt_4,149,428). coords(wk2gt_4,105,103,428).  visible(wk2gt_4,428). happensAt(walking(wk2gt_4),428).
orientation(wk2gt_2,141,429). coords(wk2gt_2,98,63,429).  visible(wk2gt_2,429). happensAt(inactive(wk2gt_2),429). orientation(wk2gt_4,149,429). coords(wk2gt_4,105,102,429).  visible(wk2gt_4,429). happensAt(walking(wk2gt_4),429).
orientation(wk2gt_2,141,430). coords(wk2gt_2,98,63,430).  visible(wk2gt_2,430). happensAt(inactive(wk2gt_2),430). orientation(wk2gt_4,149,430). coords(wk2gt_4,104,102,430).  visible(wk2gt_4,430). happensAt(walking(wk2gt_4),430).
orientation(wk2gt_2,141,431). coords(wk2gt_2,98,63,431).  visible(wk2gt_2,431). happensAt(inactive(wk2gt_2),431). orientation(wk2gt_4,149,431). coords(wk2gt_4,104,101,431).  visible(wk2gt_4,431). happensAt(walking(wk2gt_4),431).
orientation(wk2gt_2,141,432). coords(wk2gt_2,98,63,432).  visible(wk2gt_2,432). happensAt(inactive(wk2gt_2),432). orientation(wk2gt_4,149,432). coords(wk2gt_4,104,101,432).  visible(wk2gt_4,432). happensAt(walking(wk2gt_4),432).
orientation(wk2gt_2,141,433). coords(wk2gt_2,98,63,433).  visible(wk2gt_2,433). happensAt(inactive(wk2gt_2),433). orientation(wk2gt_4,149,433). coords(wk2gt_4,104,101,433).  visible(wk2gt_4,433). happensAt(walking(wk2gt_4),433).
orientation(wk2gt_2,141,434). coords(wk2gt_2,98,63,434).  visible(wk2gt_2,434). happensAt(inactive(wk2gt_2),434). orientation(wk2gt_4,149,434). coords(wk2gt_4,103,100,434).  visible(wk2gt_4,434). happensAt(walking(wk2gt_4),434).
orientation(wk2gt_2,141,435). coords(wk2gt_2,98,63,435).  visible(wk2gt_2,435). happensAt(inactive(wk2gt_2),435). orientation(wk2gt_4,149,435). coords(wk2gt_4,103,100,435).  visible(wk2gt_4,435). happensAt(walking(wk2gt_4),435).
orientation(wk2gt_2,141,436). coords(wk2gt_2,98,63,436).  visible(wk2gt_2,436). happensAt(inactive(wk2gt_2),436). orientation(wk2gt_4,149,436). coords(wk2gt_4,102,100,436).  visible(wk2gt_4,436). happensAt(walking(wk2gt_4),436).
orientation(wk2gt_2,141,437). coords(wk2gt_2,98,63,437).  visible(wk2gt_2,437). happensAt(inactive(wk2gt_2),437). orientation(wk2gt_4,149,437). coords(wk2gt_4,101,100,437).  visible(wk2gt_4,437). happensAt(walking(wk2gt_4),437).
orientation(wk2gt_2,141,438). coords(wk2gt_2,98,63,438).  visible(wk2gt_2,438). happensAt(inactive(wk2gt_2),438). orientation(wk2gt_4,149,438). coords(wk2gt_4,101,100,438).  visible(wk2gt_4,438). happensAt(walking(wk2gt_4),438).
orientation(wk2gt_2,141,439). coords(wk2gt_2,98,63,439).  visible(wk2gt_2,439). happensAt(inactive(wk2gt_2),439). orientation(wk2gt_4,149,439). coords(wk2gt_4,99,100,439).  visible(wk2gt_4,439). happensAt(walking(wk2gt_4),439).
#show holdsAt/2.