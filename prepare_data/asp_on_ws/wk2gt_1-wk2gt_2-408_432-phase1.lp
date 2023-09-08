
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

	orientation(wk2gt_1,125,408). coords(wk2gt_1,115,53,408).  visible(wk2gt_1,408). happensAt(inactive(wk2gt_1),408). orientation(wk2gt_2,141,408). coords(wk2gt_2,98,63,408).  visible(wk2gt_2,408). happensAt(inactive(wk2gt_2),408).
orientation(wk2gt_1,125,409). coords(wk2gt_1,115,53,409).  visible(wk2gt_1,409). happensAt(inactive(wk2gt_1),409). orientation(wk2gt_2,141,409). coords(wk2gt_2,98,63,409).  visible(wk2gt_2,409). happensAt(inactive(wk2gt_2),409).
orientation(wk2gt_1,125,410). coords(wk2gt_1,115,53,410).  visible(wk2gt_1,410). happensAt(inactive(wk2gt_1),410). orientation(wk2gt_2,141,410). coords(wk2gt_2,98,63,410).  visible(wk2gt_2,410). happensAt(inactive(wk2gt_2),410).
orientation(wk2gt_1,125,411). coords(wk2gt_1,115,53,411).  visible(wk2gt_1,411). happensAt(inactive(wk2gt_1),411). orientation(wk2gt_2,141,411). coords(wk2gt_2,98,63,411).  visible(wk2gt_2,411). happensAt(inactive(wk2gt_2),411).
orientation(wk2gt_1,125,412). coords(wk2gt_1,115,53,412).  visible(wk2gt_1,412). happensAt(inactive(wk2gt_1),412). orientation(wk2gt_2,141,412). coords(wk2gt_2,98,63,412).  visible(wk2gt_2,412). happensAt(inactive(wk2gt_2),412).
orientation(wk2gt_1,125,413). coords(wk2gt_1,115,53,413).  visible(wk2gt_1,413). happensAt(inactive(wk2gt_1),413). orientation(wk2gt_2,141,413). coords(wk2gt_2,98,63,413).  visible(wk2gt_2,413). happensAt(inactive(wk2gt_2),413).
orientation(wk2gt_1,125,414). coords(wk2gt_1,115,53,414).  visible(wk2gt_1,414). happensAt(inactive(wk2gt_1),414). orientation(wk2gt_2,141,414). coords(wk2gt_2,98,63,414).  visible(wk2gt_2,414). happensAt(inactive(wk2gt_2),414).
orientation(wk2gt_1,125,415). coords(wk2gt_1,115,53,415).  visible(wk2gt_1,415). happensAt(inactive(wk2gt_1),415). orientation(wk2gt_2,141,415). coords(wk2gt_2,98,63,415).  visible(wk2gt_2,415). happensAt(inactive(wk2gt_2),415).
orientation(wk2gt_1,125,416). coords(wk2gt_1,115,53,416).  visible(wk2gt_1,416). happensAt(inactive(wk2gt_1),416). orientation(wk2gt_2,141,416). coords(wk2gt_2,98,63,416).  visible(wk2gt_2,416). happensAt(inactive(wk2gt_2),416).
orientation(wk2gt_1,125,417). coords(wk2gt_1,115,53,417).  visible(wk2gt_1,417). happensAt(inactive(wk2gt_1),417). orientation(wk2gt_2,141,417). coords(wk2gt_2,98,63,417).  visible(wk2gt_2,417). happensAt(inactive(wk2gt_2),417).
orientation(wk2gt_1,125,418). coords(wk2gt_1,115,53,418).  visible(wk2gt_1,418). happensAt(inactive(wk2gt_1),418). orientation(wk2gt_2,141,418). coords(wk2gt_2,98,63,418).  visible(wk2gt_2,418). happensAt(inactive(wk2gt_2),418).
orientation(wk2gt_1,125,419). coords(wk2gt_1,115,53,419).  visible(wk2gt_1,419). happensAt(inactive(wk2gt_1),419). orientation(wk2gt_2,141,419). coords(wk2gt_2,98,63,419).  visible(wk2gt_2,419). happensAt(inactive(wk2gt_2),419).
orientation(wk2gt_1,125,420). coords(wk2gt_1,115,53,420).  visible(wk2gt_1,420). happensAt(inactive(wk2gt_1),420). orientation(wk2gt_2,141,420). coords(wk2gt_2,98,63,420).  visible(wk2gt_2,420). happensAt(inactive(wk2gt_2),420).
orientation(wk2gt_1,125,421). coords(wk2gt_1,115,53,421).  visible(wk2gt_1,421). happensAt(inactive(wk2gt_1),421). orientation(wk2gt_2,141,421). coords(wk2gt_2,98,63,421).  visible(wk2gt_2,421). happensAt(inactive(wk2gt_2),421).
orientation(wk2gt_1,125,422). coords(wk2gt_1,115,53,422).  visible(wk2gt_1,422). happensAt(inactive(wk2gt_1),422). orientation(wk2gt_2,141,422). coords(wk2gt_2,98,63,422).  visible(wk2gt_2,422). happensAt(inactive(wk2gt_2),422).
orientation(wk2gt_1,125,423). coords(wk2gt_1,115,53,423).  visible(wk2gt_1,423). happensAt(inactive(wk2gt_1),423). orientation(wk2gt_2,141,423). coords(wk2gt_2,98,63,423).  visible(wk2gt_2,423). happensAt(inactive(wk2gt_2),423).
orientation(wk2gt_1,125,424). coords(wk2gt_1,115,53,424).  visible(wk2gt_1,424). happensAt(inactive(wk2gt_1),424). orientation(wk2gt_2,141,424). coords(wk2gt_2,98,63,424).  visible(wk2gt_2,424). happensAt(inactive(wk2gt_2),424).
orientation(wk2gt_1,125,425). coords(wk2gt_1,115,53,425).  visible(wk2gt_1,425). happensAt(inactive(wk2gt_1),425). orientation(wk2gt_2,141,425). coords(wk2gt_2,98,63,425).  visible(wk2gt_2,425). happensAt(inactive(wk2gt_2),425).
orientation(wk2gt_1,125,426). coords(wk2gt_1,115,53,426).  visible(wk2gt_1,426). happensAt(inactive(wk2gt_1),426). orientation(wk2gt_2,141,426). coords(wk2gt_2,98,63,426).  visible(wk2gt_2,426). happensAt(inactive(wk2gt_2),426).
orientation(wk2gt_1,125,427). coords(wk2gt_1,115,53,427).  visible(wk2gt_1,427). happensAt(inactive(wk2gt_1),427). orientation(wk2gt_2,141,427). coords(wk2gt_2,98,63,427).  visible(wk2gt_2,427). happensAt(inactive(wk2gt_2),427).
orientation(wk2gt_1,125,428). coords(wk2gt_1,115,53,428).  visible(wk2gt_1,428). happensAt(inactive(wk2gt_1),428). orientation(wk2gt_2,141,428). coords(wk2gt_2,98,63,428).  visible(wk2gt_2,428). happensAt(inactive(wk2gt_2),428).
orientation(wk2gt_1,125,429). coords(wk2gt_1,115,53,429).  visible(wk2gt_1,429). happensAt(inactive(wk2gt_1),429). orientation(wk2gt_2,141,429). coords(wk2gt_2,98,63,429).  visible(wk2gt_2,429). happensAt(inactive(wk2gt_2),429).
orientation(wk2gt_1,125,430). coords(wk2gt_1,115,53,430).  visible(wk2gt_1,430). happensAt(inactive(wk2gt_1),430). orientation(wk2gt_2,141,430). coords(wk2gt_2,98,63,430).  visible(wk2gt_2,430). happensAt(inactive(wk2gt_2),430).
orientation(wk2gt_1,125,431). coords(wk2gt_1,115,53,431).  visible(wk2gt_1,431). happensAt(inactive(wk2gt_1),431). orientation(wk2gt_2,141,431). coords(wk2gt_2,98,63,431).  visible(wk2gt_2,431). happensAt(inactive(wk2gt_2),431).
orientation(wk2gt_1,125,432). coords(wk2gt_1,115,53,432).  visible(wk2gt_1,432). happensAt(inactive(wk2gt_1),432). orientation(wk2gt_2,141,432). coords(wk2gt_2,98,63,432).  visible(wk2gt_2,432). happensAt(inactive(wk2gt_2),432).
#show holdsAt/2.