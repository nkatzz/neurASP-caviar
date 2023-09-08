
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

	orientation(mwt1gt_1,160,413). coords(mwt1gt_1,84,112,413).  visible(mwt1gt_1,413). happensAt(walking(mwt1gt_1),413). orientation(mwt1gt_2,140,413). coords(mwt1gt_2,89,103,413).  visible(mwt1gt_2,413). happensAt(walking(mwt1gt_2),413).
orientation(mwt1gt_1,160,414). coords(mwt1gt_1,84,112,414).  visible(mwt1gt_1,414). happensAt(walking(mwt1gt_1),414). orientation(mwt1gt_2,140,414). coords(mwt1gt_2,89,103,414).  visible(mwt1gt_2,414). happensAt(walking(mwt1gt_2),414).
orientation(mwt1gt_1,160,415). coords(mwt1gt_1,84,112,415).  visible(mwt1gt_1,415). happensAt(walking(mwt1gt_1),415). orientation(mwt1gt_2,140,415). coords(mwt1gt_2,89,103,415).  visible(mwt1gt_2,415). happensAt(walking(mwt1gt_2),415).
orientation(mwt1gt_1,160,416). coords(mwt1gt_1,84,112,416).  visible(mwt1gt_1,416). happensAt(walking(mwt1gt_1),416). orientation(mwt1gt_2,140,416). coords(mwt1gt_2,89,103,416).  visible(mwt1gt_2,416). happensAt(walking(mwt1gt_2),416).
orientation(mwt1gt_1,160,417). coords(mwt1gt_1,84,112,417).  visible(mwt1gt_1,417). happensAt(walking(mwt1gt_1),417). orientation(mwt1gt_2,140,417). coords(mwt1gt_2,89,103,417).  visible(mwt1gt_2,417). happensAt(walking(mwt1gt_2),417).
orientation(mwt1gt_1,160,418). coords(mwt1gt_1,84,112,418).  visible(mwt1gt_1,418). happensAt(walking(mwt1gt_1),418). orientation(mwt1gt_2,140,418). coords(mwt1gt_2,89,103,418).  visible(mwt1gt_2,418). happensAt(walking(mwt1gt_2),418).
orientation(mwt1gt_1,160,419). coords(mwt1gt_1,84,112,419).  visible(mwt1gt_1,419). happensAt(walking(mwt1gt_1),419). orientation(mwt1gt_2,140,419). coords(mwt1gt_2,89,103,419).  visible(mwt1gt_2,419). happensAt(walking(mwt1gt_2),419).
orientation(mwt1gt_1,160,420). coords(mwt1gt_1,84,112,420).  visible(mwt1gt_1,420). happensAt(walking(mwt1gt_1),420). orientation(mwt1gt_2,140,420). coords(mwt1gt_2,89,103,420).  visible(mwt1gt_2,420). happensAt(walking(mwt1gt_2),420).
orientation(mwt1gt_1,160,421). coords(mwt1gt_1,80,109,421).  visible(mwt1gt_1,421). happensAt(walking(mwt1gt_1),421). orientation(mwt1gt_2,140,421). coords(mwt1gt_2,81,96,421).  visible(mwt1gt_2,421). happensAt(walking(mwt1gt_2),421).
orientation(mwt1gt_1,160,422). coords(mwt1gt_1,80,109,422).  visible(mwt1gt_1,422). happensAt(walking(mwt1gt_1),422). orientation(mwt1gt_2,140,422). coords(mwt1gt_2,81,96,422).  visible(mwt1gt_2,422). happensAt(walking(mwt1gt_2),422).
orientation(mwt1gt_1,160,423). coords(mwt1gt_1,80,108,423).  visible(mwt1gt_1,423). happensAt(walking(mwt1gt_1),423). orientation(mwt1gt_2,140,423). coords(mwt1gt_2,81,95,423).  visible(mwt1gt_2,423). happensAt(walking(mwt1gt_2),423).
orientation(mwt1gt_1,160,424). coords(mwt1gt_1,80,108,424).  visible(mwt1gt_1,424). happensAt(walking(mwt1gt_1),424). orientation(mwt1gt_2,140,424). coords(mwt1gt_2,81,95,424).  visible(mwt1gt_2,424). happensAt(walking(mwt1gt_2),424).
orientation(mwt1gt_1,160,425). coords(mwt1gt_1,80,108,425).  visible(mwt1gt_1,425). happensAt(walking(mwt1gt_1),425). orientation(mwt1gt_2,140,425). coords(mwt1gt_2,81,95,425).  visible(mwt1gt_2,425). happensAt(walking(mwt1gt_2),425).
orientation(mwt1gt_1,160,426). coords(mwt1gt_1,80,108,426).  visible(mwt1gt_1,426). happensAt(walking(mwt1gt_1),426). orientation(mwt1gt_2,140,426). coords(mwt1gt_2,81,95,426).  visible(mwt1gt_2,426). happensAt(walking(mwt1gt_2),426).
orientation(mwt1gt_1,160,427). coords(mwt1gt_1,79,106,427).  visible(mwt1gt_1,427). happensAt(walking(mwt1gt_1),427). orientation(mwt1gt_2,140,427). coords(mwt1gt_2,80,94,427).  visible(mwt1gt_2,427). happensAt(walking(mwt1gt_2),427).
orientation(mwt1gt_1,160,428). coords(mwt1gt_1,79,106,428).  visible(mwt1gt_1,428). happensAt(walking(mwt1gt_1),428). orientation(mwt1gt_2,140,428). coords(mwt1gt_2,80,94,428).  visible(mwt1gt_2,428). happensAt(walking(mwt1gt_2),428).
orientation(mwt1gt_1,160,429). coords(mwt1gt_1,79,106,429).  visible(mwt1gt_1,429). happensAt(walking(mwt1gt_1),429). orientation(mwt1gt_2,140,429). coords(mwt1gt_2,80,94,429).  visible(mwt1gt_2,429). happensAt(walking(mwt1gt_2),429).
orientation(mwt1gt_1,160,430). coords(mwt1gt_1,79,106,430).  visible(mwt1gt_1,430). happensAt(walking(mwt1gt_1),430). orientation(mwt1gt_2,140,430). coords(mwt1gt_2,80,94,430).  visible(mwt1gt_2,430). happensAt(walking(mwt1gt_2),430).
orientation(mwt1gt_1,160,431). coords(mwt1gt_1,78,105,431).  visible(mwt1gt_1,431). happensAt(walking(mwt1gt_1),431). orientation(mwt1gt_2,140,431). coords(mwt1gt_2,80,94,431).  visible(mwt1gt_2,431). happensAt(walking(mwt1gt_2),431).
orientation(mwt1gt_1,160,432). coords(mwt1gt_1,78,105,432).  visible(mwt1gt_1,432). happensAt(walking(mwt1gt_1),432). orientation(mwt1gt_2,140,432). coords(mwt1gt_2,80,94,432).  visible(mwt1gt_2,432). happensAt(walking(mwt1gt_2),432).
orientation(mwt1gt_1,160,433). coords(mwt1gt_1,78,105,433).  visible(mwt1gt_1,433). happensAt(walking(mwt1gt_1),433). orientation(mwt1gt_2,140,433). coords(mwt1gt_2,80,94,433).  visible(mwt1gt_2,433). happensAt(walking(mwt1gt_2),433).
orientation(mwt1gt_1,160,434). coords(mwt1gt_1,74,104,434).  visible(mwt1gt_1,434). happensAt(walking(mwt1gt_1),434). orientation(mwt1gt_2,140,434). coords(mwt1gt_2,75,92,434).  visible(mwt1gt_2,434). happensAt(walking(mwt1gt_2),434).
orientation(mwt1gt_1,160,435). coords(mwt1gt_1,74,104,435).  visible(mwt1gt_1,435). happensAt(walking(mwt1gt_1),435). orientation(mwt1gt_2,140,435). coords(mwt1gt_2,75,92,435).  visible(mwt1gt_2,435). happensAt(walking(mwt1gt_2),435).
orientation(mwt1gt_1,160,436). coords(mwt1gt_1,74,104,436).  visible(mwt1gt_1,436). happensAt(walking(mwt1gt_1),436). orientation(mwt1gt_2,140,436). coords(mwt1gt_2,75,92,436).  visible(mwt1gt_2,436). happensAt(walking(mwt1gt_2),436).
orientation(mwt1gt_1,160,437). coords(mwt1gt_1,74,104,437).  visible(mwt1gt_1,437). happensAt(walking(mwt1gt_1),437). orientation(mwt1gt_2,140,437). coords(mwt1gt_2,75,92,437).  visible(mwt1gt_2,437). happensAt(walking(mwt1gt_2),437).
#show holdsAt/2.