
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

	orientation(fomdgt3_2,42,414). coords(fomdgt3_2,262,96,414).  visible(fomdgt3_2,414). happensAt(inactive(fomdgt3_2),414). orientation(fomdgt3_6,140,414). coords(fomdgt3_6,64,83,414).  visible(fomdgt3_6,414). happensAt(running(fomdgt3_6),414).
orientation(fomdgt3_2,42,415). coords(fomdgt3_2,262,96,415).  visible(fomdgt3_2,415). happensAt(inactive(fomdgt3_2),415). orientation(fomdgt3_6,140,415). coords(fomdgt3_6,63,84,415).  visible(fomdgt3_6,415). happensAt(running(fomdgt3_6),415).
orientation(fomdgt3_2,42,416). coords(fomdgt3_2,262,96,416).  visible(fomdgt3_2,416). happensAt(inactive(fomdgt3_2),416). orientation(fomdgt3_6,140,416). coords(fomdgt3_6,62,84,416).  visible(fomdgt3_6,416). happensAt(running(fomdgt3_6),416).
orientation(fomdgt3_2,42,417). coords(fomdgt3_2,262,96,417).  visible(fomdgt3_2,417). happensAt(inactive(fomdgt3_2),417). orientation(fomdgt3_6,140,417). coords(fomdgt3_6,62,84,417).  visible(fomdgt3_6,417). happensAt(running(fomdgt3_6),417).
orientation(fomdgt3_2,42,418). coords(fomdgt3_2,262,96,418).  visible(fomdgt3_2,418). happensAt(inactive(fomdgt3_2),418). orientation(fomdgt3_6,140,418). coords(fomdgt3_6,60,85,418).  visible(fomdgt3_6,418). happensAt(running(fomdgt3_6),418).
orientation(fomdgt3_2,42,419). coords(fomdgt3_2,262,96,419).  visible(fomdgt3_2,419). happensAt(inactive(fomdgt3_2),419). orientation(fomdgt3_6,140,419). coords(fomdgt3_6,60,88,419).  visible(fomdgt3_6,419). happensAt(running(fomdgt3_6),419).
orientation(fomdgt3_2,42,420). coords(fomdgt3_2,262,96,420).  visible(fomdgt3_2,420). happensAt(inactive(fomdgt3_2),420). orientation(fomdgt3_6,140,420). coords(fomdgt3_6,60,88,420).  visible(fomdgt3_6,420). happensAt(running(fomdgt3_6),420).
orientation(fomdgt3_2,42,421). coords(fomdgt3_2,262,96,421).  visible(fomdgt3_2,421). happensAt(inactive(fomdgt3_2),421). orientation(fomdgt3_6,140,421). coords(fomdgt3_6,60,89,421).  visible(fomdgt3_6,421). happensAt(walking(fomdgt3_6),421).
orientation(fomdgt3_2,42,422). coords(fomdgt3_2,262,96,422).  visible(fomdgt3_2,422). happensAt(inactive(fomdgt3_2),422). orientation(fomdgt3_6,140,422). coords(fomdgt3_6,59,89,422).  visible(fomdgt3_6,422). happensAt(walking(fomdgt3_6),422).
orientation(fomdgt3_2,42,423). coords(fomdgt3_2,262,96,423).  visible(fomdgt3_2,423). happensAt(inactive(fomdgt3_2),423). orientation(fomdgt3_6,140,423). coords(fomdgt3_6,58,89,423).  visible(fomdgt3_6,423). happensAt(walking(fomdgt3_6),423).
orientation(fomdgt3_2,42,424). coords(fomdgt3_2,262,96,424).  visible(fomdgt3_2,424). happensAt(inactive(fomdgt3_2),424). orientation(fomdgt3_6,160,424). coords(fomdgt3_6,54,90,424).  visible(fomdgt3_6,424). happensAt(walking(fomdgt3_6),424).
orientation(fomdgt3_2,42,425). coords(fomdgt3_2,262,96,425).  visible(fomdgt3_2,425). happensAt(inactive(fomdgt3_2),425). orientation(fomdgt3_6,160,425). coords(fomdgt3_6,53,91,425).  visible(fomdgt3_6,425). happensAt(walking(fomdgt3_6),425).
orientation(fomdgt3_2,42,426). coords(fomdgt3_2,262,96,426).  visible(fomdgt3_2,426). happensAt(inactive(fomdgt3_2),426). orientation(fomdgt3_6,160,426). coords(fomdgt3_6,53,91,426).  visible(fomdgt3_6,426). happensAt(walking(fomdgt3_6),426).
orientation(fomdgt3_2,42,427). coords(fomdgt3_2,262,96,427).  visible(fomdgt3_2,427). happensAt(inactive(fomdgt3_2),427). orientation(fomdgt3_6,160,427). coords(fomdgt3_6,51,91,427).  visible(fomdgt3_6,427). happensAt(walking(fomdgt3_6),427).
orientation(fomdgt3_2,42,428). coords(fomdgt3_2,262,96,428).  visible(fomdgt3_2,428). happensAt(inactive(fomdgt3_2),428). orientation(fomdgt3_6,160,428). coords(fomdgt3_6,51,92,428).  visible(fomdgt3_6,428). happensAt(walking(fomdgt3_6),428).
orientation(fomdgt3_2,42,429). coords(fomdgt3_2,262,96,429).  visible(fomdgt3_2,429). happensAt(inactive(fomdgt3_2),429). orientation(fomdgt3_6,160,429). coords(fomdgt3_6,51,93,429).  visible(fomdgt3_6,429). happensAt(walking(fomdgt3_6),429).
orientation(fomdgt3_2,42,430). coords(fomdgt3_2,262,96,430).  visible(fomdgt3_2,430). happensAt(inactive(fomdgt3_2),430). orientation(fomdgt3_6,160,430). coords(fomdgt3_6,51,93,430).  visible(fomdgt3_6,430). happensAt(walking(fomdgt3_6),430).
orientation(fomdgt3_2,42,431). coords(fomdgt3_2,262,96,431).  visible(fomdgt3_2,431). happensAt(inactive(fomdgt3_2),431). orientation(fomdgt3_6,160,431). coords(fomdgt3_6,50,93,431).  visible(fomdgt3_6,431). happensAt(walking(fomdgt3_6),431).
orientation(fomdgt3_2,42,432). coords(fomdgt3_2,262,96,432).  visible(fomdgt3_2,432). happensAt(inactive(fomdgt3_2),432). orientation(fomdgt3_6,160,432). coords(fomdgt3_6,49,93,432).  visible(fomdgt3_6,432). happensAt(walking(fomdgt3_6),432).
orientation(fomdgt3_2,42,433). coords(fomdgt3_2,262,96,433).  visible(fomdgt3_2,433). happensAt(inactive(fomdgt3_2),433). orientation(fomdgt3_6,160,433). coords(fomdgt3_6,49,94,433).  visible(fomdgt3_6,433). happensAt(walking(fomdgt3_6),433).
orientation(fomdgt3_2,42,434). coords(fomdgt3_2,262,96,434).  visible(fomdgt3_2,434). happensAt(inactive(fomdgt3_2),434). orientation(fomdgt3_6,160,434). coords(fomdgt3_6,49,94,434).  visible(fomdgt3_6,434). happensAt(walking(fomdgt3_6),434).
orientation(fomdgt3_2,42,435). coords(fomdgt3_2,262,96,435).  visible(fomdgt3_2,435). happensAt(inactive(fomdgt3_2),435). orientation(fomdgt3_6,160,435). coords(fomdgt3_6,49,94,435).  visible(fomdgt3_6,435). happensAt(walking(fomdgt3_6),435).
orientation(fomdgt3_2,42,436). coords(fomdgt3_2,262,96,436).  visible(fomdgt3_2,436). happensAt(inactive(fomdgt3_2),436). orientation(fomdgt3_6,160,436). coords(fomdgt3_6,49,95,436).  visible(fomdgt3_6,436). happensAt(walking(fomdgt3_6),436).
orientation(fomdgt3_2,42,437). coords(fomdgt3_2,262,96,437).  visible(fomdgt3_2,437). happensAt(inactive(fomdgt3_2),437). orientation(fomdgt3_6,160,437). coords(fomdgt3_6,49,97,437).  visible(fomdgt3_6,437). happensAt(walking(fomdgt3_6),437).
orientation(fomdgt3_2,42,438). coords(fomdgt3_2,262,96,438).  visible(fomdgt3_2,438). happensAt(inactive(fomdgt3_2),438). orientation(fomdgt3_6,160,438). coords(fomdgt3_6,49,97,438).  visible(fomdgt3_6,438). happensAt(walking(fomdgt3_6),438).
#show holdsAt/2.