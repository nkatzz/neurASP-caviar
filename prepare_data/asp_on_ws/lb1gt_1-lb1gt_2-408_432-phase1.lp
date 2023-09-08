
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

	orientation(lb1gt_1,151,408). coords(lb1gt_1,166,168,408).  visible(lb1gt_1,408). happensAt(active(lb1gt_1),408). orientation(lb1gt_2,144,408). coords(lb1gt_2,189,141,408).  visible(lb1gt_2,408). happensAt(active(lb1gt_2),408).
orientation(lb1gt_1,151,409). coords(lb1gt_1,166,168,409).  visible(lb1gt_1,409). happensAt(active(lb1gt_1),409). orientation(lb1gt_2,144,409). coords(lb1gt_2,189,141,409).  visible(lb1gt_2,409). happensAt(active(lb1gt_2),409).
orientation(lb1gt_1,151,410). coords(lb1gt_1,166,168,410).  visible(lb1gt_1,410). happensAt(active(lb1gt_1),410). orientation(lb1gt_2,144,410). coords(lb1gt_2,189,141,410).  visible(lb1gt_2,410). happensAt(active(lb1gt_2),410).
orientation(lb1gt_1,151,411). coords(lb1gt_1,167,168,411).  visible(lb1gt_1,411). happensAt(active(lb1gt_1),411). orientation(lb1gt_2,144,411). coords(lb1gt_2,189,141,411).  visible(lb1gt_2,411). happensAt(active(lb1gt_2),411).
orientation(lb1gt_1,151,412). coords(lb1gt_1,167,168,412).  visible(lb1gt_1,412). happensAt(active(lb1gt_1),412). orientation(lb1gt_2,144,412). coords(lb1gt_2,189,141,412).  visible(lb1gt_2,412). happensAt(active(lb1gt_2),412).
orientation(lb1gt_1,151,413). coords(lb1gt_1,167,168,413).  visible(lb1gt_1,413). happensAt(active(lb1gt_1),413). orientation(lb1gt_2,144,413). coords(lb1gt_2,189,141,413).  visible(lb1gt_2,413). happensAt(active(lb1gt_2),413).
orientation(lb1gt_1,151,414). coords(lb1gt_1,167,170,414).  visible(lb1gt_1,414). happensAt(active(lb1gt_1),414). orientation(lb1gt_2,144,414). coords(lb1gt_2,189,141,414).  visible(lb1gt_2,414). happensAt(active(lb1gt_2),414).
orientation(lb1gt_1,151,415). coords(lb1gt_1,167,170,415).  visible(lb1gt_1,415). happensAt(active(lb1gt_1),415). orientation(lb1gt_2,144,415). coords(lb1gt_2,189,141,415).  visible(lb1gt_2,415). happensAt(active(lb1gt_2),415).
orientation(lb1gt_1,151,416). coords(lb1gt_1,167,172,416).  visible(lb1gt_1,416). happensAt(active(lb1gt_1),416). orientation(lb1gt_2,144,416). coords(lb1gt_2,189,141,416).  visible(lb1gt_2,416). happensAt(active(lb1gt_2),416).
orientation(lb1gt_1,151,417). coords(lb1gt_1,167,173,417).  visible(lb1gt_1,417). happensAt(active(lb1gt_1),417). orientation(lb1gt_2,144,417). coords(lb1gt_2,189,141,417).  visible(lb1gt_2,417). happensAt(active(lb1gt_2),417).
orientation(lb1gt_1,151,418). coords(lb1gt_1,167,173,418).  visible(lb1gt_1,418). happensAt(active(lb1gt_1),418). orientation(lb1gt_2,144,418). coords(lb1gt_2,189,141,418).  visible(lb1gt_2,418). happensAt(active(lb1gt_2),418).
orientation(lb1gt_1,151,419). coords(lb1gt_1,166,174,419).  visible(lb1gt_1,419). happensAt(active(lb1gt_1),419). orientation(lb1gt_2,144,419). coords(lb1gt_2,189,141,419).  visible(lb1gt_2,419). happensAt(active(lb1gt_2),419).
orientation(lb1gt_1,151,420). coords(lb1gt_1,166,175,420).  visible(lb1gt_1,420). happensAt(walking(lb1gt_1),420). orientation(lb1gt_2,144,420). coords(lb1gt_2,189,141,420).  visible(lb1gt_2,420). happensAt(active(lb1gt_2),420).
orientation(lb1gt_1,151,421). coords(lb1gt_1,166,176,421).  visible(lb1gt_1,421). happensAt(walking(lb1gt_1),421). orientation(lb1gt_2,144,421). coords(lb1gt_2,189,141,421).  visible(lb1gt_2,421). happensAt(active(lb1gt_2),421).
orientation(lb1gt_1,151,422). coords(lb1gt_1,166,177,422).  visible(lb1gt_1,422). happensAt(walking(lb1gt_1),422). orientation(lb1gt_2,144,422). coords(lb1gt_2,189,141,422).  visible(lb1gt_2,422). happensAt(active(lb1gt_2),422).
orientation(lb1gt_1,151,423). coords(lb1gt_1,166,178,423).  visible(lb1gt_1,423). happensAt(walking(lb1gt_1),423). orientation(lb1gt_2,144,423). coords(lb1gt_2,189,141,423).  visible(lb1gt_2,423). happensAt(active(lb1gt_2),423).
orientation(lb1gt_1,151,424). coords(lb1gt_1,166,179,424).  visible(lb1gt_1,424). happensAt(walking(lb1gt_1),424). orientation(lb1gt_2,144,424). coords(lb1gt_2,189,141,424).  visible(lb1gt_2,424). happensAt(active(lb1gt_2),424).
orientation(lb1gt_1,161,425). coords(lb1gt_1,166,179,425).  visible(lb1gt_1,425). happensAt(walking(lb1gt_1),425). orientation(lb1gt_2,144,425). coords(lb1gt_2,189,141,425).  visible(lb1gt_2,425). happensAt(active(lb1gt_2),425).
orientation(lb1gt_1,162,426). coords(lb1gt_1,166,180,426).  visible(lb1gt_1,426). happensAt(walking(lb1gt_1),426). orientation(lb1gt_2,144,426). coords(lb1gt_2,189,141,426).  visible(lb1gt_2,426). happensAt(active(lb1gt_2),426).
orientation(lb1gt_1,162,427). coords(lb1gt_1,166,180,427).  visible(lb1gt_1,427). happensAt(walking(lb1gt_1),427). orientation(lb1gt_2,144,427). coords(lb1gt_2,189,141,427).  visible(lb1gt_2,427). happensAt(active(lb1gt_2),427).
orientation(lb1gt_1,163,428). coords(lb1gt_1,165,181,428).  visible(lb1gt_1,428). happensAt(walking(lb1gt_1),428). orientation(lb1gt_2,144,428). coords(lb1gt_2,189,141,428).  visible(lb1gt_2,428). happensAt(active(lb1gt_2),428).
orientation(lb1gt_1,162,429). coords(lb1gt_1,165,181,429).  visible(lb1gt_1,429). happensAt(walking(lb1gt_1),429). orientation(lb1gt_2,144,429). coords(lb1gt_2,189,141,429).  visible(lb1gt_2,429). happensAt(active(lb1gt_2),429).
orientation(lb1gt_1,162,430). coords(lb1gt_1,165,182,430).  visible(lb1gt_1,430). happensAt(walking(lb1gt_1),430). orientation(lb1gt_2,144,430). coords(lb1gt_2,189,141,430).  visible(lb1gt_2,430). happensAt(active(lb1gt_2),430).
orientation(lb1gt_1,163,431). coords(lb1gt_1,165,182,431).  visible(lb1gt_1,431). happensAt(walking(lb1gt_1),431). orientation(lb1gt_2,144,431). coords(lb1gt_2,189,141,431).  visible(lb1gt_2,431). happensAt(active(lb1gt_2),431).
orientation(lb1gt_1,162,432). coords(lb1gt_1,165,183,432).  visible(lb1gt_1,432). happensAt(walking(lb1gt_1),432). orientation(lb1gt_2,144,432). coords(lb1gt_2,189,141,432).  visible(lb1gt_2,432). happensAt(active(lb1gt_2),432).
#show holdsAt/2.