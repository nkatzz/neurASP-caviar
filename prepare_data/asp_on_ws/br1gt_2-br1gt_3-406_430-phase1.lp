
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

	orientation(br1gt_2,7,406). coords(br1gt_2,84,208,406).  visible(br1gt_2,406). happensAt(inactive(br1gt_2),406). orientation(br1gt_3,162,406). coords(br1gt_3,44,156,406).  visible(br1gt_3,406). happensAt(walking(br1gt_3),406).
orientation(br1gt_2,7,407). coords(br1gt_2,84,208,407).  visible(br1gt_2,407). happensAt(inactive(br1gt_2),407). orientation(br1gt_3,162,407). coords(br1gt_3,45,156,407).  visible(br1gt_3,407). happensAt(walking(br1gt_3),407).
orientation(br1gt_2,7,408). coords(br1gt_2,84,208,408).  visible(br1gt_2,408). happensAt(inactive(br1gt_2),408). orientation(br1gt_3,162,408). coords(br1gt_3,45,156,408).  visible(br1gt_3,408). happensAt(walking(br1gt_3),408).
orientation(br1gt_2,7,409). coords(br1gt_2,84,208,409).  visible(br1gt_2,409). happensAt(inactive(br1gt_2),409). orientation(br1gt_3,162,409). coords(br1gt_3,46,156,409).  visible(br1gt_3,409). happensAt(walking(br1gt_3),409).
orientation(br1gt_2,7,410). coords(br1gt_2,84,208,410).  visible(br1gt_2,410). happensAt(inactive(br1gt_2),410). orientation(br1gt_3,162,410). coords(br1gt_3,46,156,410).  visible(br1gt_3,410). happensAt(walking(br1gt_3),410).
orientation(br1gt_2,10,411). coords(br1gt_2,84,208,411).  visible(br1gt_2,411). happensAt(inactive(br1gt_2),411). orientation(br1gt_3,162,411). coords(br1gt_3,47,156,411).  visible(br1gt_3,411). happensAt(walking(br1gt_3),411).
orientation(br1gt_2,10,412). coords(br1gt_2,84,208,412).  visible(br1gt_2,412). happensAt(inactive(br1gt_2),412). orientation(br1gt_3,162,412). coords(br1gt_3,47,156,412).  visible(br1gt_3,412). happensAt(walking(br1gt_3),412).
orientation(br1gt_2,10,413). coords(br1gt_2,84,208,413).  visible(br1gt_2,413). happensAt(inactive(br1gt_2),413). orientation(br1gt_3,162,413). coords(br1gt_3,49,156,413).  visible(br1gt_3,413). happensAt(walking(br1gt_3),413).
orientation(br1gt_2,10,414). coords(br1gt_2,84,208,414).  visible(br1gt_2,414). happensAt(inactive(br1gt_2),414). orientation(br1gt_3,162,414). coords(br1gt_3,50,156,414).  visible(br1gt_3,414). happensAt(walking(br1gt_3),414).
orientation(br1gt_2,10,415). coords(br1gt_2,85,208,415).  visible(br1gt_2,415). happensAt(inactive(br1gt_2),415). orientation(br1gt_3,162,415). coords(br1gt_3,50,156,415).  visible(br1gt_3,415). happensAt(walking(br1gt_3),415).
orientation(br1gt_2,10,416). coords(br1gt_2,85,208,416).  visible(br1gt_2,416). happensAt(inactive(br1gt_2),416). orientation(br1gt_3,162,416). coords(br1gt_3,50,156,416).  visible(br1gt_3,416). happensAt(walking(br1gt_3),416).
orientation(br1gt_2,10,417). coords(br1gt_2,85,208,417).  visible(br1gt_2,417). happensAt(inactive(br1gt_2),417). orientation(br1gt_3,162,417). coords(br1gt_3,50,156,417).  visible(br1gt_3,417). happensAt(walking(br1gt_3),417).
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
#show holdsAt/2.