
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

	orientation(fomdgt3_6,140,398). coords(fomdgt3_6,85,81,398).  visible(fomdgt3_6,398). happensAt(running(fomdgt3_6),398). orientation(fomdgt3_7,134,398). coords(fomdgt3_7,79,99,398).  visible(fomdgt3_7,398). happensAt(walking(fomdgt3_7),398).
orientation(fomdgt3_6,140,399). coords(fomdgt3_6,83,81,399).  visible(fomdgt3_6,399). happensAt(running(fomdgt3_6),399). orientation(fomdgt3_7,134,399). coords(fomdgt3_7,79,99,399).  visible(fomdgt3_7,399). happensAt(walking(fomdgt3_7),399).
orientation(fomdgt3_6,140,400). coords(fomdgt3_6,82,81,400).  visible(fomdgt3_6,400). happensAt(running(fomdgt3_6),400). orientation(fomdgt3_7,134,400). coords(fomdgt3_7,79,101,400).  visible(fomdgt3_7,400). happensAt(walking(fomdgt3_7),400).
orientation(fomdgt3_6,140,401). coords(fomdgt3_6,79,81,401).  visible(fomdgt3_6,401). happensAt(running(fomdgt3_6),401). orientation(fomdgt3_7,134,401). coords(fomdgt3_7,79,101,401).  visible(fomdgt3_7,401). happensAt(walking(fomdgt3_7),401).
orientation(fomdgt3_6,140,402). coords(fomdgt3_6,79,81,402).  visible(fomdgt3_6,402). happensAt(running(fomdgt3_6),402). orientation(fomdgt3_7,134,402). coords(fomdgt3_7,80,101,402).  visible(fomdgt3_7,402). happensAt(walking(fomdgt3_7),402).
orientation(fomdgt3_6,140,403). coords(fomdgt3_6,77,81,403).  visible(fomdgt3_6,403). happensAt(running(fomdgt3_6),403). orientation(fomdgt3_7,134,403). coords(fomdgt3_7,80,103,403).  visible(fomdgt3_7,403). happensAt(walking(fomdgt3_7),403).
orientation(fomdgt3_6,140,404). coords(fomdgt3_6,76,81,404).  visible(fomdgt3_6,404). happensAt(running(fomdgt3_6),404). orientation(fomdgt3_7,134,404). coords(fomdgt3_7,80,103,404).  visible(fomdgt3_7,404). happensAt(walking(fomdgt3_7),404).
orientation(fomdgt3_6,140,405). coords(fomdgt3_6,75,81,405).  visible(fomdgt3_6,405). happensAt(running(fomdgt3_6),405). orientation(fomdgt3_7,134,405). coords(fomdgt3_7,80,103,405).  visible(fomdgt3_7,405). happensAt(walking(fomdgt3_7),405).
orientation(fomdgt3_6,140,406). coords(fomdgt3_6,74,81,406).  visible(fomdgt3_6,406). happensAt(running(fomdgt3_6),406). orientation(fomdgt3_7,134,406). coords(fomdgt3_7,80,103,406).  visible(fomdgt3_7,406). happensAt(walking(fomdgt3_7),406).
orientation(fomdgt3_6,140,407). coords(fomdgt3_6,71,82,407).  visible(fomdgt3_6,407). happensAt(running(fomdgt3_6),407). orientation(fomdgt3_7,134,407). coords(fomdgt3_7,80,104,407).  visible(fomdgt3_7,407). happensAt(walking(fomdgt3_7),407).
orientation(fomdgt3_6,140,408). coords(fomdgt3_6,71,82,408).  visible(fomdgt3_6,408). happensAt(running(fomdgt3_6),408). orientation(fomdgt3_7,134,408). coords(fomdgt3_7,80,106,408).  visible(fomdgt3_7,408). happensAt(walking(fomdgt3_7),408).
orientation(fomdgt3_6,140,409). coords(fomdgt3_6,69,82,409).  visible(fomdgt3_6,409). happensAt(running(fomdgt3_6),409). orientation(fomdgt3_7,134,409). coords(fomdgt3_7,80,108,409).  visible(fomdgt3_7,409). happensAt(walking(fomdgt3_7),409).
orientation(fomdgt3_6,140,410). coords(fomdgt3_6,68,82,410).  visible(fomdgt3_6,410). happensAt(running(fomdgt3_6),410). orientation(fomdgt3_7,134,410). coords(fomdgt3_7,80,109,410).  visible(fomdgt3_7,410). happensAt(walking(fomdgt3_7),410).
orientation(fomdgt3_6,140,411). coords(fomdgt3_6,68,82,411).  visible(fomdgt3_6,411). happensAt(running(fomdgt3_6),411). orientation(fomdgt3_7,134,411). coords(fomdgt3_7,80,110,411).  visible(fomdgt3_7,411). happensAt(walking(fomdgt3_7),411).
orientation(fomdgt3_6,140,412). coords(fomdgt3_6,67,83,412).  visible(fomdgt3_6,412). happensAt(running(fomdgt3_6),412). orientation(fomdgt3_7,134,412). coords(fomdgt3_7,80,111,412).  visible(fomdgt3_7,412). happensAt(walking(fomdgt3_7),412).
orientation(fomdgt3_6,140,413). coords(fomdgt3_6,66,83,413).  visible(fomdgt3_6,413). happensAt(running(fomdgt3_6),413). orientation(fomdgt3_7,134,413). coords(fomdgt3_7,80,112,413).  visible(fomdgt3_7,413). happensAt(walking(fomdgt3_7),413).
orientation(fomdgt3_6,140,414). coords(fomdgt3_6,64,83,414).  visible(fomdgt3_6,414). happensAt(running(fomdgt3_6),414). orientation(fomdgt3_7,134,414). coords(fomdgt3_7,80,113,414).  visible(fomdgt3_7,414). happensAt(walking(fomdgt3_7),414).
orientation(fomdgt3_6,140,415). coords(fomdgt3_6,63,84,415).  visible(fomdgt3_6,415). happensAt(running(fomdgt3_6),415). orientation(fomdgt3_7,134,415). coords(fomdgt3_7,80,115,415).  visible(fomdgt3_7,415). happensAt(walking(fomdgt3_7),415).
orientation(fomdgt3_6,140,416). coords(fomdgt3_6,62,84,416).  visible(fomdgt3_6,416). happensAt(running(fomdgt3_6),416). orientation(fomdgt3_7,134,416). coords(fomdgt3_7,80,116,416).  visible(fomdgt3_7,416). happensAt(walking(fomdgt3_7),416).
orientation(fomdgt3_6,140,417). coords(fomdgt3_6,62,84,417).  visible(fomdgt3_6,417). happensAt(running(fomdgt3_6),417). orientation(fomdgt3_7,134,417). coords(fomdgt3_7,80,116,417).  visible(fomdgt3_7,417). happensAt(walking(fomdgt3_7),417).
orientation(fomdgt3_6,140,418). coords(fomdgt3_6,60,85,418).  visible(fomdgt3_6,418). happensAt(running(fomdgt3_6),418). orientation(fomdgt3_7,134,418). coords(fomdgt3_7,80,116,418).  visible(fomdgt3_7,418). happensAt(walking(fomdgt3_7),418).
orientation(fomdgt3_6,140,419). coords(fomdgt3_6,60,88,419).  visible(fomdgt3_6,419). happensAt(running(fomdgt3_6),419). orientation(fomdgt3_7,150,419). coords(fomdgt3_7,80,116,419).  visible(fomdgt3_7,419). happensAt(walking(fomdgt3_7),419).
orientation(fomdgt3_6,140,420). coords(fomdgt3_6,60,88,420).  visible(fomdgt3_6,420). happensAt(running(fomdgt3_6),420). orientation(fomdgt3_7,150,420). coords(fomdgt3_7,80,116,420).  visible(fomdgt3_7,420). happensAt(walking(fomdgt3_7),420).
orientation(fomdgt3_6,140,421). coords(fomdgt3_6,60,89,421).  visible(fomdgt3_6,421). happensAt(walking(fomdgt3_6),421). orientation(fomdgt3_7,150,421). coords(fomdgt3_7,80,116,421).  visible(fomdgt3_7,421). happensAt(walking(fomdgt3_7),421).
orientation(fomdgt3_6,140,422). coords(fomdgt3_6,59,89,422).  visible(fomdgt3_6,422). happensAt(walking(fomdgt3_6),422). orientation(fomdgt3_7,150,422). coords(fomdgt3_7,81,116,422).  visible(fomdgt3_7,422). happensAt(walking(fomdgt3_7),422).
#show holdsAt/2.