
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

	orientation(br1gt_2,10,418). coords(br1gt_2,85,208,418).  visible(br1gt_2,418). happensAt(inactive(br1gt_2),418). orientation(br1gt_3,162,418). coords(br1gt_3,50,155,418).  visible(br1gt_3,418). happensAt(walking(br1gt_3),418).
orientation(br1gt_2,10,419). coords(br1gt_2,86,208,419).  visible(br1gt_2,419). happensAt(inactive(br1gt_2),419). orientation(br1gt_3,162,419). coords(br1gt_3,50,155,419).  visible(br1gt_3,419). happensAt(walking(br1gt_3),419).
orientation(br1gt_2,10,420). coords(br1gt_2,86,208,420).  visible(br1gt_2,420). happensAt(inactive(br1gt_2),420). orientation(br1gt_3,162,420). coords(br1gt_3,50,155,420).  visible(br1gt_3,420). happensAt(walking(br1gt_3),420).
orientation(br1gt_2,10,421). coords(br1gt_2,86,208,421).  visible(br1gt_2,421). happensAt(inactive(br1gt_2),421). orientation(br1gt_3,162,421). coords(br1gt_3,51,155,421).  visible(br1gt_3,421). happensAt(walking(br1gt_3),421).
orientation(br1gt_2,10,422). coords(br1gt_2,86,208,422).  visible(br1gt_2,422). happensAt(inactive(br1gt_2),422). orientation(br1gt_3,162,422). coords(br1gt_3,53,154,422).  visible(br1gt_3,422). happensAt(walking(br1gt_3),422).
orientation(br1gt_2,12,423). coords(br1gt_2,86,208,423).  visible(br1gt_2,423). happensAt(inactive(br1gt_2),423). orientation(br1gt_3,162,423). coords(br1gt_3,53,154,423).  visible(br1gt_3,423). happensAt(walking(br1gt_3),423).
orientation(br1gt_2,12,424). coords(br1gt_2,86,208,424).  visible(br1gt_2,424). happensAt(inactive(br1gt_2),424). orientation(br1gt_3,162,424). coords(br1gt_3,53,154,424).  visible(br1gt_3,424). happensAt(walking(br1gt_3),424).
orientation(br1gt_2,12,425). coords(br1gt_2,86,208,425).  visible(br1gt_2,425). happensAt(inactive(br1gt_2),425). orientation(br1gt_3,162,425). coords(br1gt_3,54,154,425).  visible(br1gt_3,425). happensAt(walking(br1gt_3),425).
orientation(br1gt_2,12,426). coords(br1gt_2,86,208,426).  visible(br1gt_2,426). happensAt(inactive(br1gt_2),426). orientation(br1gt_3,162,426). coords(br1gt_3,54,154,426).  visible(br1gt_3,426). happensAt(walking(br1gt_3),426).
orientation(br1gt_2,12,427). coords(br1gt_2,86,208,427).  visible(br1gt_2,427). happensAt(inactive(br1gt_2),427). orientation(br1gt_3,162,427). coords(br1gt_3,55,154,427).  visible(br1gt_3,427). happensAt(walking(br1gt_3),427).
orientation(br1gt_2,12,428). coords(br1gt_2,86,208,428).  visible(br1gt_2,428). happensAt(inactive(br1gt_2),428). orientation(br1gt_3,162,428). coords(br1gt_3,55,154,428).  visible(br1gt_3,428). happensAt(walking(br1gt_3),428).
orientation(br1gt_2,12,429). coords(br1gt_2,86,208,429).  visible(br1gt_2,429). happensAt(inactive(br1gt_2),429). orientation(br1gt_3,162,429). coords(br1gt_3,56,154,429).  visible(br1gt_3,429). happensAt(walking(br1gt_3),429).
orientation(br1gt_2,12,430). coords(br1gt_2,86,208,430).  visible(br1gt_2,430). happensAt(inactive(br1gt_2),430). orientation(br1gt_3,162,430). coords(br1gt_3,56,154,430).  visible(br1gt_3,430). happensAt(walking(br1gt_3),430).
orientation(br1gt_2,12,431). coords(br1gt_2,86,208,431).  visible(br1gt_2,431). happensAt(inactive(br1gt_2),431). orientation(br1gt_3,162,431). coords(br1gt_3,56,154,431).  visible(br1gt_3,431). happensAt(walking(br1gt_3),431).
orientation(br1gt_2,12,432). coords(br1gt_2,86,208,432).  visible(br1gt_2,432). happensAt(inactive(br1gt_2),432). orientation(br1gt_3,162,432). coords(br1gt_3,56,153,432).  visible(br1gt_3,432). happensAt(walking(br1gt_3),432).
orientation(br1gt_2,12,433). coords(br1gt_2,86,208,433).  visible(br1gt_2,433). happensAt(inactive(br1gt_2),433). orientation(br1gt_3,162,433). coords(br1gt_3,57,153,433).  visible(br1gt_3,433). happensAt(walking(br1gt_3),433).
orientation(br1gt_2,12,434). coords(br1gt_2,86,208,434).  visible(br1gt_2,434). happensAt(inactive(br1gt_2),434). orientation(br1gt_3,162,434). coords(br1gt_3,59,153,434).  visible(br1gt_3,434). happensAt(walking(br1gt_3),434).
orientation(br1gt_2,12,435). coords(br1gt_2,86,208,435).  visible(br1gt_2,435). happensAt(inactive(br1gt_2),435). orientation(br1gt_3,162,435). coords(br1gt_3,59,153,435).  visible(br1gt_3,435). happensAt(walking(br1gt_3),435).
orientation(br1gt_2,12,436). coords(br1gt_2,86,208,436).  visible(br1gt_2,436). happensAt(inactive(br1gt_2),436). orientation(br1gt_3,162,436). coords(br1gt_3,59,153,436).  visible(br1gt_3,436). happensAt(walking(br1gt_3),436).
orientation(br1gt_2,12,437). coords(br1gt_2,86,208,437).  visible(br1gt_2,437). happensAt(inactive(br1gt_2),437). orientation(br1gt_3,162,437). coords(br1gt_3,60,153,437).  visible(br1gt_3,437). happensAt(walking(br1gt_3),437).
orientation(br1gt_2,12,438). coords(br1gt_2,86,208,438).  visible(br1gt_2,438). happensAt(inactive(br1gt_2),438). orientation(br1gt_3,162,438). coords(br1gt_3,61,153,438).  visible(br1gt_3,438). happensAt(walking(br1gt_3),438).
orientation(br1gt_2,12,439). coords(br1gt_2,86,208,439).  visible(br1gt_2,439). happensAt(inactive(br1gt_2),439). orientation(br1gt_3,162,439). coords(br1gt_3,62,153,439).  visible(br1gt_3,439). happensAt(walking(br1gt_3),439).
orientation(br1gt_2,12,440). coords(br1gt_2,86,208,440).  visible(br1gt_2,440). happensAt(inactive(br1gt_2),440). orientation(br1gt_3,162,440). coords(br1gt_3,63,152,440).  visible(br1gt_3,440). happensAt(walking(br1gt_3),440).
orientation(br1gt_2,12,441). coords(br1gt_2,86,208,441).  visible(br1gt_2,441). happensAt(inactive(br1gt_2),441). orientation(br1gt_3,162,441). coords(br1gt_3,63,152,441).  visible(br1gt_3,441). happensAt(walking(br1gt_3),441).
orientation(br1gt_2,12,442). coords(br1gt_2,86,208,442).  visible(br1gt_2,442). happensAt(inactive(br1gt_2),442). orientation(br1gt_3,162,442). coords(br1gt_3,63,151,442).  visible(br1gt_3,442). happensAt(walking(br1gt_3),442).
#show holdsAt/2.