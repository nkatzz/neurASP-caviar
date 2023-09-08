
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

	orientation(fra1gt_6,173,414). coords(fra1gt_6,90,128,414).  visible(fra1gt_6,414). happensAt(running(fra1gt_6),414). orientation(fra1gt_7,146,414). coords(fra1gt_7,85,79,414).  visible(fra1gt_7,414). happensAt(running(fra1gt_7),414).
orientation(fra1gt_6,173,415). coords(fra1gt_6,88,129,415).  visible(fra1gt_6,415). happensAt(running(fra1gt_6),415). orientation(fra1gt_7,146,415). coords(fra1gt_7,84,79,415).  visible(fra1gt_7,415). happensAt(running(fra1gt_7),415).
orientation(fra1gt_6,173,416). coords(fra1gt_6,86,131,416).  visible(fra1gt_6,416). happensAt(running(fra1gt_6),416). orientation(fra1gt_7,146,416). coords(fra1gt_7,82,79,416).  visible(fra1gt_7,416). happensAt(running(fra1gt_7),416).
orientation(fra1gt_6,173,417). coords(fra1gt_6,85,132,417).  visible(fra1gt_6,417). happensAt(running(fra1gt_6),417). orientation(fra1gt_7,146,417). coords(fra1gt_7,82,79,417).  visible(fra1gt_7,417). happensAt(running(fra1gt_7),417).
orientation(fra1gt_6,173,418). coords(fra1gt_6,84,134,418).  visible(fra1gt_6,418). happensAt(running(fra1gt_6),418). orientation(fra1gt_7,146,418). coords(fra1gt_7,80,79,418).  visible(fra1gt_7,418). happensAt(running(fra1gt_7),418).
orientation(fra1gt_6,173,419). coords(fra1gt_6,84,136,419).  visible(fra1gt_6,419). happensAt(running(fra1gt_6),419). orientation(fra1gt_7,146,419). coords(fra1gt_7,79,79,419).  visible(fra1gt_7,419). happensAt(running(fra1gt_7),419).
orientation(fra1gt_6,173,420). coords(fra1gt_6,82,138,420).  visible(fra1gt_6,420). happensAt(running(fra1gt_6),420). orientation(fra1gt_7,146,420). coords(fra1gt_7,79,80,420).  visible(fra1gt_7,420). happensAt(running(fra1gt_7),420).
orientation(fra1gt_6,173,421). coords(fra1gt_6,79,139,421).  visible(fra1gt_6,421). happensAt(running(fra1gt_6),421). orientation(fra1gt_7,146,421). coords(fra1gt_7,78,79,421).  visible(fra1gt_7,421). happensAt(running(fra1gt_7),421).
orientation(fra1gt_6,173,422). coords(fra1gt_6,78,141,422).  visible(fra1gt_6,422). happensAt(running(fra1gt_6),422). orientation(fra1gt_7,146,422). coords(fra1gt_7,78,80,422).  visible(fra1gt_7,422). happensAt(running(fra1gt_7),422).
orientation(fra1gt_6,173,423). coords(fra1gt_6,76,142,423).  visible(fra1gt_6,423). happensAt(running(fra1gt_6),423). orientation(fra1gt_7,146,423). coords(fra1gt_7,76,80,423).  visible(fra1gt_7,423). happensAt(running(fra1gt_7),423).
orientation(fra1gt_6,173,424). coords(fra1gt_6,73,142,424).  visible(fra1gt_6,424). happensAt(running(fra1gt_6),424). orientation(fra1gt_7,146,424). coords(fra1gt_7,76,80,424).  visible(fra1gt_7,424). happensAt(running(fra1gt_7),424).
orientation(fra1gt_6,173,425). coords(fra1gt_6,72,143,425).  visible(fra1gt_6,425). happensAt(running(fra1gt_6),425). orientation(fra1gt_7,146,425). coords(fra1gt_7,75,80,425).  visible(fra1gt_7,425). happensAt(running(fra1gt_7),425).
orientation(fra1gt_6,177,426). coords(fra1gt_6,71,144,426).  visible(fra1gt_6,426). happensAt(running(fra1gt_6),426). orientation(fra1gt_7,146,426). coords(fra1gt_7,74,81,426).  visible(fra1gt_7,426). happensAt(running(fra1gt_7),426).
orientation(fra1gt_6,5,427). coords(fra1gt_6,70,143,427).  visible(fra1gt_6,427). happensAt(running(fra1gt_6),427). orientation(fra1gt_7,146,427). coords(fra1gt_7,72,81,427).  visible(fra1gt_7,427). happensAt(running(fra1gt_7),427).
orientation(fra1gt_6,5,428). coords(fra1gt_6,69,144,428).  visible(fra1gt_6,428). happensAt(running(fra1gt_6),428). orientation(fra1gt_7,146,428). coords(fra1gt_7,71,80,428).  visible(fra1gt_7,428). happensAt(running(fra1gt_7),428).
orientation(fra1gt_6,5,429). coords(fra1gt_6,68,147,429).  visible(fra1gt_6,429). happensAt(running(fra1gt_6),429). orientation(fra1gt_7,146,429). coords(fra1gt_7,70,81,429).  visible(fra1gt_7,429). happensAt(running(fra1gt_7),429).
orientation(fra1gt_6,5,430). coords(fra1gt_6,66,149,430).  visible(fra1gt_6,430). happensAt(running(fra1gt_6),430). orientation(fra1gt_7,146,430). coords(fra1gt_7,69,81,430).  visible(fra1gt_7,430). happensAt(running(fra1gt_7),430).
orientation(fra1gt_6,5,431). coords(fra1gt_6,65,150,431).  visible(fra1gt_6,431). happensAt(running(fra1gt_6),431). orientation(fra1gt_7,146,431). coords(fra1gt_7,68,81,431).  visible(fra1gt_7,431). happensAt(running(fra1gt_7),431).
orientation(fra1gt_6,5,432). coords(fra1gt_6,63,152,432).  visible(fra1gt_6,432). happensAt(running(fra1gt_6),432). orientation(fra1gt_7,146,432). coords(fra1gt_7,68,82,432).  visible(fra1gt_7,432). happensAt(running(fra1gt_7),432).
orientation(fra1gt_6,5,433). coords(fra1gt_6,62,153,433).  visible(fra1gt_6,433). happensAt(running(fra1gt_6),433). orientation(fra1gt_7,146,433). coords(fra1gt_7,68,82,433).  visible(fra1gt_7,433). happensAt(running(fra1gt_7),433).
orientation(fra1gt_6,5,434). coords(fra1gt_6,61,154,434).  visible(fra1gt_6,434). happensAt(running(fra1gt_6),434). orientation(fra1gt_7,146,434). coords(fra1gt_7,66,82,434).  visible(fra1gt_7,434). happensAt(running(fra1gt_7),434).
orientation(fra1gt_6,177,435). coords(fra1gt_6,59,154,435).  visible(fra1gt_6,435). happensAt(running(fra1gt_6),435). orientation(fra1gt_7,146,435). coords(fra1gt_7,66,83,435).  visible(fra1gt_7,435). happensAt(running(fra1gt_7),435).
orientation(fra1gt_6,3,436). coords(fra1gt_6,58,156,436).  visible(fra1gt_6,436). happensAt(running(fra1gt_6),436). orientation(fra1gt_7,146,436). coords(fra1gt_7,64,83,436).  visible(fra1gt_7,436). happensAt(running(fra1gt_7),436).
orientation(fra1gt_6,4,437). coords(fra1gt_6,58,157,437).  visible(fra1gt_6,437). happensAt(running(fra1gt_6),437). orientation(fra1gt_7,146,437). coords(fra1gt_7,63,84,437).  visible(fra1gt_7,437). happensAt(walking(fra1gt_7),437).
orientation(fra1gt_6,179,438). coords(fra1gt_6,57,160,438).  visible(fra1gt_6,438). happensAt(running(fra1gt_6),438). orientation(fra1gt_7,146,438). coords(fra1gt_7,63,84,438).  visible(fra1gt_7,438). happensAt(walking(fra1gt_7),438).
#show holdsAt/2.