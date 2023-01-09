
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

	orientation(mwt2gt_1,150,409). coords(mwt2gt_1,101,90,409).  visible(mwt2gt_1,409). happensAt(walking(mwt2gt_1),409). orientation(mwt2gt_2,149,409). coords(mwt2gt_2,96,115,409).  visible(mwt2gt_2,409). happensAt(walking(mwt2gt_2),409).
orientation(mwt2gt_1,150,410). coords(mwt2gt_1,100,90,410).  visible(mwt2gt_1,410). happensAt(walking(mwt2gt_1),410). orientation(mwt2gt_2,149,410). coords(mwt2gt_2,95,116,410).  visible(mwt2gt_2,410). happensAt(walking(mwt2gt_2),410).
orientation(mwt2gt_1,150,411). coords(mwt2gt_1,99,90,411).  visible(mwt2gt_1,411). happensAt(walking(mwt2gt_1),411). orientation(mwt2gt_2,149,411). coords(mwt2gt_2,93,116,411).  visible(mwt2gt_2,411). happensAt(walking(mwt2gt_2),411).
orientation(mwt2gt_1,150,412). coords(mwt2gt_1,99,90,412).  visible(mwt2gt_1,412). happensAt(walking(mwt2gt_1),412). orientation(mwt2gt_2,149,412). coords(mwt2gt_2,93,118,412).  visible(mwt2gt_2,412). happensAt(walking(mwt2gt_2),412).
orientation(mwt2gt_1,150,413). coords(mwt2gt_1,98,90,413).  visible(mwt2gt_1,413). happensAt(walking(mwt2gt_1),413). orientation(mwt2gt_2,149,413). coords(mwt2gt_2,93,119,413).  visible(mwt2gt_2,413). happensAt(walking(mwt2gt_2),413).
orientation(mwt2gt_1,150,414). coords(mwt2gt_1,96,90,414).  visible(mwt2gt_1,414). happensAt(walking(mwt2gt_1),414). orientation(mwt2gt_2,149,414). coords(mwt2gt_2,93,119,414).  visible(mwt2gt_2,414). happensAt(walking(mwt2gt_2),414).
orientation(mwt2gt_1,150,415). coords(mwt2gt_1,96,90,415).  visible(mwt2gt_1,415). happensAt(walking(mwt2gt_1),415). orientation(mwt2gt_2,149,415). coords(mwt2gt_2,93,119,415).  visible(mwt2gt_2,415). happensAt(walking(mwt2gt_2),415).
orientation(mwt2gt_1,150,416). coords(mwt2gt_1,95,90,416).  visible(mwt2gt_1,416). happensAt(walking(mwt2gt_1),416). orientation(mwt2gt_2,149,416). coords(mwt2gt_2,93,119,416).  visible(mwt2gt_2,416). happensAt(walking(mwt2gt_2),416).
orientation(mwt2gt_1,150,417). coords(mwt2gt_1,95,90,417).  visible(mwt2gt_1,417). happensAt(walking(mwt2gt_1),417). orientation(mwt2gt_2,149,417). coords(mwt2gt_2,92,119,417).  visible(mwt2gt_2,417). happensAt(walking(mwt2gt_2),417).
orientation(mwt2gt_1,150,418). coords(mwt2gt_1,95,90,418).  visible(mwt2gt_1,418). happensAt(walking(mwt2gt_1),418). orientation(mwt2gt_2,149,418). coords(mwt2gt_2,91,120,418).  visible(mwt2gt_2,418). happensAt(walking(mwt2gt_2),418).
orientation(mwt2gt_1,150,419). coords(mwt2gt_1,94,90,419).  visible(mwt2gt_1,419). happensAt(walking(mwt2gt_1),419). orientation(mwt2gt_2,149,419). coords(mwt2gt_2,90,120,419).  visible(mwt2gt_2,419). happensAt(walking(mwt2gt_2),419).
orientation(mwt2gt_1,150,420). coords(mwt2gt_1,94,89,420).  visible(mwt2gt_1,420). happensAt(walking(mwt2gt_1),420). orientation(mwt2gt_2,149,420). coords(mwt2gt_2,90,122,420).  visible(mwt2gt_2,420). happensAt(walking(mwt2gt_2),420).
orientation(mwt2gt_1,150,421). coords(mwt2gt_1,94,89,421).  visible(mwt2gt_1,421). happensAt(walking(mwt2gt_1),421). orientation(mwt2gt_2,149,421). coords(mwt2gt_2,90,122,421).  visible(mwt2gt_2,421). happensAt(walking(mwt2gt_2),421).
orientation(mwt2gt_1,150,422). coords(mwt2gt_1,94,89,422).  visible(mwt2gt_1,422). happensAt(walking(mwt2gt_1),422). orientation(mwt2gt_2,149,422). coords(mwt2gt_2,88,122,422).  visible(mwt2gt_2,422). happensAt(walking(mwt2gt_2),422).
orientation(mwt2gt_1,151,423). coords(mwt2gt_1,93,89,423).  visible(mwt2gt_1,423). happensAt(walking(mwt2gt_1),423). orientation(mwt2gt_2,149,423). coords(mwt2gt_2,87,122,423).  visible(mwt2gt_2,423). happensAt(walking(mwt2gt_2),423).
orientation(mwt2gt_1,153,424). coords(mwt2gt_1,92,89,424).  visible(mwt2gt_1,424). happensAt(walking(mwt2gt_1),424). orientation(mwt2gt_2,149,424). coords(mwt2gt_2,87,122,424).  visible(mwt2gt_2,424). happensAt(walking(mwt2gt_2),424).
orientation(mwt2gt_1,153,425). coords(mwt2gt_1,91,90,425).  visible(mwt2gt_1,425). happensAt(walking(mwt2gt_1),425). orientation(mwt2gt_2,149,425). coords(mwt2gt_2,86,122,425).  visible(mwt2gt_2,425). happensAt(walking(mwt2gt_2),425).
orientation(mwt2gt_1,153,426). coords(mwt2gt_1,90,90,426).  visible(mwt2gt_1,426). happensAt(walking(mwt2gt_1),426). orientation(mwt2gt_2,149,426). coords(mwt2gt_2,85,122,426).  visible(mwt2gt_2,426). happensAt(walking(mwt2gt_2),426).
orientation(mwt2gt_1,153,427). coords(mwt2gt_1,90,90,427).  visible(mwt2gt_1,427). happensAt(walking(mwt2gt_1),427). orientation(mwt2gt_2,149,427). coords(mwt2gt_2,85,122,427).  visible(mwt2gt_2,427). happensAt(walking(mwt2gt_2),427).
orientation(mwt2gt_1,153,428). coords(mwt2gt_1,89,90,428).  visible(mwt2gt_1,428). happensAt(walking(mwt2gt_1),428). orientation(mwt2gt_2,150,428). coords(mwt2gt_2,85,122,428).  visible(mwt2gt_2,428). happensAt(walking(mwt2gt_2),428).
orientation(mwt2gt_1,153,429). coords(mwt2gt_1,89,90,429).  visible(mwt2gt_1,429). happensAt(walking(mwt2gt_1),429). orientation(mwt2gt_2,150,429). coords(mwt2gt_2,85,122,429).  visible(mwt2gt_2,429). happensAt(walking(mwt2gt_2),429).
orientation(mwt2gt_1,153,430). coords(mwt2gt_1,89,91,430).  visible(mwt2gt_1,430). happensAt(walking(mwt2gt_1),430). orientation(mwt2gt_2,150,430). coords(mwt2gt_2,85,122,430).  visible(mwt2gt_2,430). happensAt(walking(mwt2gt_2),430).
orientation(mwt2gt_1,153,431). coords(mwt2gt_1,89,91,431).  visible(mwt2gt_1,431). happensAt(walking(mwt2gt_1),431). orientation(mwt2gt_2,150,431). coords(mwt2gt_2,85,122,431).  visible(mwt2gt_2,431). happensAt(walking(mwt2gt_2),431).
orientation(mwt2gt_1,153,432). coords(mwt2gt_1,88,92,432).  visible(mwt2gt_1,432). happensAt(walking(mwt2gt_1),432). orientation(mwt2gt_2,150,432). coords(mwt2gt_2,85,122,432).  visible(mwt2gt_2,432). happensAt(walking(mwt2gt_2),432).
orientation(mwt2gt_1,153,433). coords(mwt2gt_1,87,92,433).  visible(mwt2gt_1,433). happensAt(walking(mwt2gt_1),433). orientation(mwt2gt_2,150,433). coords(mwt2gt_2,84,122,433).  visible(mwt2gt_2,433). happensAt(walking(mwt2gt_2),433).
#show holdsAt/2.