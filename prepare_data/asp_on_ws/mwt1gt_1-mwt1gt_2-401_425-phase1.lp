
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

	orientation(mwt1gt_1,160,401). coords(mwt1gt_1,90,118,401).  visible(mwt1gt_1,401). happensAt(walking(mwt1gt_1),401). orientation(mwt1gt_2,140,401). coords(mwt1gt_2,90,104,401).  visible(mwt1gt_2,401). happensAt(walking(mwt1gt_2),401).
orientation(mwt1gt_1,160,402). coords(mwt1gt_1,90,118,402).  visible(mwt1gt_1,402). happensAt(walking(mwt1gt_1),402). orientation(mwt1gt_2,140,402). coords(mwt1gt_2,90,104,402).  visible(mwt1gt_2,402). happensAt(walking(mwt1gt_2),402).
orientation(mwt1gt_1,160,403). coords(mwt1gt_1,90,118,403).  visible(mwt1gt_1,403). happensAt(walking(mwt1gt_1),403). orientation(mwt1gt_2,140,403). coords(mwt1gt_2,90,104,403).  visible(mwt1gt_2,403). happensAt(walking(mwt1gt_2),403).
orientation(mwt1gt_1,160,404). coords(mwt1gt_1,89,116,404).  visible(mwt1gt_1,404). happensAt(walking(mwt1gt_1),404). orientation(mwt1gt_2,140,404). coords(mwt1gt_2,89,103,404).  visible(mwt1gt_2,404). happensAt(walking(mwt1gt_2),404).
orientation(mwt1gt_1,160,405). coords(mwt1gt_1,89,116,405).  visible(mwt1gt_1,405). happensAt(walking(mwt1gt_1),405). orientation(mwt1gt_2,140,405). coords(mwt1gt_2,89,103,405).  visible(mwt1gt_2,405). happensAt(walking(mwt1gt_2),405).
orientation(mwt1gt_1,160,406). coords(mwt1gt_1,89,116,406).  visible(mwt1gt_1,406). happensAt(walking(mwt1gt_1),406). orientation(mwt1gt_2,140,406). coords(mwt1gt_2,89,103,406).  visible(mwt1gt_2,406). happensAt(walking(mwt1gt_2),406).
orientation(mwt1gt_1,160,407). coords(mwt1gt_1,89,116,407).  visible(mwt1gt_1,407). happensAt(walking(mwt1gt_1),407). orientation(mwt1gt_2,140,407). coords(mwt1gt_2,89,103,407).  visible(mwt1gt_2,407). happensAt(walking(mwt1gt_2),407).
orientation(mwt1gt_1,160,408). coords(mwt1gt_1,85,113,408).  visible(mwt1gt_1,408). happensAt(walking(mwt1gt_1),408). orientation(mwt1gt_2,140,408). coords(mwt1gt_2,89,103,408).  visible(mwt1gt_2,408). happensAt(walking(mwt1gt_2),408).
orientation(mwt1gt_1,160,409). coords(mwt1gt_1,85,113,409).  visible(mwt1gt_1,409). happensAt(walking(mwt1gt_1),409). orientation(mwt1gt_2,140,409). coords(mwt1gt_2,89,103,409).  visible(mwt1gt_2,409). happensAt(walking(mwt1gt_2),409).
orientation(mwt1gt_1,160,410). coords(mwt1gt_1,85,113,410).  visible(mwt1gt_1,410). happensAt(walking(mwt1gt_1),410). orientation(mwt1gt_2,140,410). coords(mwt1gt_2,89,103,410).  visible(mwt1gt_2,410). happensAt(walking(mwt1gt_2),410).
orientation(mwt1gt_1,160,411). coords(mwt1gt_1,85,113,411).  visible(mwt1gt_1,411). happensAt(walking(mwt1gt_1),411). orientation(mwt1gt_2,140,411). coords(mwt1gt_2,89,103,411).  visible(mwt1gt_2,411). happensAt(walking(mwt1gt_2),411).
orientation(mwt1gt_1,160,412). coords(mwt1gt_1,84,112,412).  visible(mwt1gt_1,412). happensAt(walking(mwt1gt_1),412). orientation(mwt1gt_2,140,412). coords(mwt1gt_2,89,103,412).  visible(mwt1gt_2,412). happensAt(walking(mwt1gt_2),412).
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
#show holdsAt/2.