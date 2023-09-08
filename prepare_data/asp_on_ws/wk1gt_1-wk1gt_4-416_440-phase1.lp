
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

	orientation(wk1gt_1,146,416). coords(wk1gt_1,98,101,416).  visible(wk1gt_1,416). happensAt(walking(wk1gt_1),416). orientation(wk1gt_4,163,416). coords(wk1gt_4,53,176,416).  visible(wk1gt_4,416). happensAt(active(wk1gt_4),416).
orientation(wk1gt_1,146,417). coords(wk1gt_1,98,101,417).  visible(wk1gt_1,417). happensAt(walking(wk1gt_1),417). orientation(wk1gt_4,160,417). coords(wk1gt_4,53,175,417).  visible(wk1gt_4,417). happensAt(active(wk1gt_4),417).
orientation(wk1gt_1,146,418). coords(wk1gt_1,97,101,418).  visible(wk1gt_1,418). happensAt(walking(wk1gt_1),418). orientation(wk1gt_4,159,418). coords(wk1gt_4,53,175,418).  visible(wk1gt_4,418). happensAt(active(wk1gt_4),418).
orientation(wk1gt_1,146,419). coords(wk1gt_1,97,100,419).  visible(wk1gt_1,419). happensAt(walking(wk1gt_1),419). orientation(wk1gt_4,157,419). coords(wk1gt_4,54,175,419).  visible(wk1gt_4,419). happensAt(active(wk1gt_4),419).
orientation(wk1gt_1,146,420). coords(wk1gt_1,97,100,420).  visible(wk1gt_1,420). happensAt(walking(wk1gt_1),420). orientation(wk1gt_4,157,420). coords(wk1gt_4,54,175,420).  visible(wk1gt_4,420). happensAt(active(wk1gt_4),420).
orientation(wk1gt_1,146,421). coords(wk1gt_1,96,99,421).  visible(wk1gt_1,421). happensAt(walking(wk1gt_1),421). orientation(wk1gt_4,157,421). coords(wk1gt_4,54,175,421).  visible(wk1gt_4,421). happensAt(active(wk1gt_4),421).
orientation(wk1gt_1,146,422). coords(wk1gt_1,95,98,422).  visible(wk1gt_1,422). happensAt(walking(wk1gt_1),422). orientation(wk1gt_4,157,422). coords(wk1gt_4,54,174,422).  visible(wk1gt_4,422). happensAt(active(wk1gt_4),422).
orientation(wk1gt_1,153,423). coords(wk1gt_1,94,98,423).  visible(wk1gt_1,423). happensAt(walking(wk1gt_1),423). orientation(wk1gt_4,157,423). coords(wk1gt_4,54,174,423).  visible(wk1gt_4,423). happensAt(active(wk1gt_4),423).
orientation(wk1gt_1,153,424). coords(wk1gt_1,94,98,424).  visible(wk1gt_1,424). happensAt(walking(wk1gt_1),424). orientation(wk1gt_4,157,424). coords(wk1gt_4,54,174,424).  visible(wk1gt_4,424). happensAt(active(wk1gt_4),424).
orientation(wk1gt_1,153,425). coords(wk1gt_1,94,98,425).  visible(wk1gt_1,425). happensAt(walking(wk1gt_1),425). orientation(wk1gt_4,157,425). coords(wk1gt_4,54,173,425).  visible(wk1gt_4,425). happensAt(active(wk1gt_4),425).
orientation(wk1gt_1,153,426). coords(wk1gt_1,94,98,426).  visible(wk1gt_1,426). happensAt(walking(wk1gt_1),426). orientation(wk1gt_4,147,426). coords(wk1gt_4,54,173,426).  visible(wk1gt_4,426). happensAt(active(wk1gt_4),426).
orientation(wk1gt_1,150,427). coords(wk1gt_1,94,98,427).  visible(wk1gt_1,427). happensAt(walking(wk1gt_1),427). orientation(wk1gt_4,147,427). coords(wk1gt_4,54,172,427).  visible(wk1gt_4,427). happensAt(active(wk1gt_4),427).
orientation(wk1gt_1,150,428). coords(wk1gt_1,94,98,428).  visible(wk1gt_1,428). happensAt(walking(wk1gt_1),428). orientation(wk1gt_4,147,428). coords(wk1gt_4,54,172,428).  visible(wk1gt_4,428). happensAt(active(wk1gt_4),428).
orientation(wk1gt_1,150,429). coords(wk1gt_1,93,97,429).  visible(wk1gt_1,429). happensAt(walking(wk1gt_1),429). orientation(wk1gt_4,147,429). coords(wk1gt_4,54,172,429).  visible(wk1gt_4,429). happensAt(active(wk1gt_4),429).
orientation(wk1gt_1,148,430). coords(wk1gt_1,93,97,430).  visible(wk1gt_1,430). happensAt(walking(wk1gt_1),430). orientation(wk1gt_4,147,430). coords(wk1gt_4,54,171,430).  visible(wk1gt_4,430). happensAt(active(wk1gt_4),430).
orientation(wk1gt_1,148,431). coords(wk1gt_1,93,97,431).  visible(wk1gt_1,431). happensAt(walking(wk1gt_1),431). orientation(wk1gt_4,147,431). coords(wk1gt_4,54,171,431).  visible(wk1gt_4,431). happensAt(active(wk1gt_4),431).
orientation(wk1gt_1,148,432). coords(wk1gt_1,91,97,432).  visible(wk1gt_1,432). happensAt(walking(wk1gt_1),432). orientation(wk1gt_4,147,432). coords(wk1gt_4,54,171,432).  visible(wk1gt_4,432). happensAt(active(wk1gt_4),432).
orientation(wk1gt_1,148,433). coords(wk1gt_1,91,95,433).  visible(wk1gt_1,433). happensAt(walking(wk1gt_1),433). orientation(wk1gt_4,147,433). coords(wk1gt_4,54,171,433).  visible(wk1gt_4,433). happensAt(active(wk1gt_4),433).
orientation(wk1gt_1,148,434). coords(wk1gt_1,91,95,434).  visible(wk1gt_1,434). happensAt(walking(wk1gt_1),434). orientation(wk1gt_4,147,434). coords(wk1gt_4,54,170,434).  visible(wk1gt_4,434). happensAt(active(wk1gt_4),434).
orientation(wk1gt_1,148,435). coords(wk1gt_1,91,95,435).  visible(wk1gt_1,435). happensAt(walking(wk1gt_1),435). orientation(wk1gt_4,147,435). coords(wk1gt_4,55,170,435).  visible(wk1gt_4,435). happensAt(active(wk1gt_4),435).
orientation(wk1gt_1,148,436). coords(wk1gt_1,91,94,436).  visible(wk1gt_1,436). happensAt(walking(wk1gt_1),436). orientation(wk1gt_4,147,436). coords(wk1gt_4,55,169,436).  visible(wk1gt_4,436). happensAt(active(wk1gt_4),436).
orientation(wk1gt_1,148,437). coords(wk1gt_1,91,94,437).  visible(wk1gt_1,437). happensAt(walking(wk1gt_1),437). orientation(wk1gt_4,147,437). coords(wk1gt_4,56,169,437).  visible(wk1gt_4,437). happensAt(active(wk1gt_4),437).
orientation(wk1gt_1,148,438). coords(wk1gt_1,90,93,438).  visible(wk1gt_1,438). happensAt(walking(wk1gt_1),438). orientation(wk1gt_4,147,438). coords(wk1gt_4,56,169,438).  visible(wk1gt_4,438). happensAt(active(wk1gt_4),438).
orientation(wk1gt_1,148,439). coords(wk1gt_1,89,92,439).  visible(wk1gt_1,439). happensAt(walking(wk1gt_1),439). orientation(wk1gt_4,147,439). coords(wk1gt_4,56,169,439).  visible(wk1gt_4,439). happensAt(active(wk1gt_4),439).
orientation(wk1gt_1,148,440). coords(wk1gt_1,89,92,440).  visible(wk1gt_1,440). happensAt(walking(wk1gt_1),440). orientation(wk1gt_4,147,440). coords(wk1gt_4,56,169,440).  visible(wk1gt_4,440). happensAt(active(wk1gt_4),440).
#show holdsAt/2.