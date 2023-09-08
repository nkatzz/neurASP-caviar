
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

	orientation(fomdgt3_1,150,398). coords(fomdgt3_1,46,102,398).  visible(fomdgt3_1,398). happensAt(inactive(fomdgt3_1),398). orientation(fomdgt3_7,134,398). coords(fomdgt3_7,79,99,398).  visible(fomdgt3_7,398). happensAt(walking(fomdgt3_7),398).
orientation(fomdgt3_1,150,399). coords(fomdgt3_1,46,102,399).  visible(fomdgt3_1,399). happensAt(inactive(fomdgt3_1),399). orientation(fomdgt3_7,134,399). coords(fomdgt3_7,79,99,399).  visible(fomdgt3_7,399). happensAt(walking(fomdgt3_7),399).
orientation(fomdgt3_1,150,400). coords(fomdgt3_1,46,102,400).  visible(fomdgt3_1,400). happensAt(inactive(fomdgt3_1),400). orientation(fomdgt3_7,134,400). coords(fomdgt3_7,79,101,400).  visible(fomdgt3_7,400). happensAt(walking(fomdgt3_7),400).
orientation(fomdgt3_1,150,401). coords(fomdgt3_1,46,102,401).  visible(fomdgt3_1,401). happensAt(inactive(fomdgt3_1),401). orientation(fomdgt3_7,134,401). coords(fomdgt3_7,79,101,401).  visible(fomdgt3_7,401). happensAt(walking(fomdgt3_7),401).
orientation(fomdgt3_1,150,402). coords(fomdgt3_1,46,102,402).  visible(fomdgt3_1,402). happensAt(inactive(fomdgt3_1),402). orientation(fomdgt3_7,134,402). coords(fomdgt3_7,80,101,402).  visible(fomdgt3_7,402). happensAt(walking(fomdgt3_7),402).
orientation(fomdgt3_1,150,403). coords(fomdgt3_1,46,102,403).  visible(fomdgt3_1,403). happensAt(inactive(fomdgt3_1),403). orientation(fomdgt3_7,134,403). coords(fomdgt3_7,80,103,403).  visible(fomdgt3_7,403). happensAt(walking(fomdgt3_7),403).
orientation(fomdgt3_1,150,404). coords(fomdgt3_1,46,102,404).  visible(fomdgt3_1,404). happensAt(inactive(fomdgt3_1),404). orientation(fomdgt3_7,134,404). coords(fomdgt3_7,80,103,404).  visible(fomdgt3_7,404). happensAt(walking(fomdgt3_7),404).
orientation(fomdgt3_1,150,405). coords(fomdgt3_1,46,102,405).  visible(fomdgt3_1,405). happensAt(inactive(fomdgt3_1),405). orientation(fomdgt3_7,134,405). coords(fomdgt3_7,80,103,405).  visible(fomdgt3_7,405). happensAt(walking(fomdgt3_7),405).
orientation(fomdgt3_1,150,406). coords(fomdgt3_1,46,102,406).  visible(fomdgt3_1,406). happensAt(inactive(fomdgt3_1),406). orientation(fomdgt3_7,134,406). coords(fomdgt3_7,80,103,406).  visible(fomdgt3_7,406). happensAt(walking(fomdgt3_7),406).
orientation(fomdgt3_1,150,407). coords(fomdgt3_1,46,102,407).  visible(fomdgt3_1,407). happensAt(inactive(fomdgt3_1),407). orientation(fomdgt3_7,134,407). coords(fomdgt3_7,80,104,407).  visible(fomdgt3_7,407). happensAt(walking(fomdgt3_7),407).
orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_7,134,408). coords(fomdgt3_7,80,106,408).  visible(fomdgt3_7,408). happensAt(walking(fomdgt3_7),408).
orientation(fomdgt3_1,150,409). coords(fomdgt3_1,46,102,409).  visible(fomdgt3_1,409). happensAt(inactive(fomdgt3_1),409). orientation(fomdgt3_7,134,409). coords(fomdgt3_7,80,108,409).  visible(fomdgt3_7,409). happensAt(walking(fomdgt3_7),409).
orientation(fomdgt3_1,150,410). coords(fomdgt3_1,46,102,410).  visible(fomdgt3_1,410). happensAt(inactive(fomdgt3_1),410). orientation(fomdgt3_7,134,410). coords(fomdgt3_7,80,109,410).  visible(fomdgt3_7,410). happensAt(walking(fomdgt3_7),410).
orientation(fomdgt3_1,150,411). coords(fomdgt3_1,46,102,411).  visible(fomdgt3_1,411). happensAt(inactive(fomdgt3_1),411). orientation(fomdgt3_7,134,411). coords(fomdgt3_7,80,110,411).  visible(fomdgt3_7,411). happensAt(walking(fomdgt3_7),411).
orientation(fomdgt3_1,150,412). coords(fomdgt3_1,46,102,412).  visible(fomdgt3_1,412). happensAt(inactive(fomdgt3_1),412). orientation(fomdgt3_7,134,412). coords(fomdgt3_7,80,111,412).  visible(fomdgt3_7,412). happensAt(walking(fomdgt3_7),412).
orientation(fomdgt3_1,150,413). coords(fomdgt3_1,46,102,413).  visible(fomdgt3_1,413). happensAt(inactive(fomdgt3_1),413). orientation(fomdgt3_7,134,413). coords(fomdgt3_7,80,112,413).  visible(fomdgt3_7,413). happensAt(walking(fomdgt3_7),413).
orientation(fomdgt3_1,150,414). coords(fomdgt3_1,46,102,414).  visible(fomdgt3_1,414). happensAt(inactive(fomdgt3_1),414). orientation(fomdgt3_7,134,414). coords(fomdgt3_7,80,113,414).  visible(fomdgt3_7,414). happensAt(walking(fomdgt3_7),414).
orientation(fomdgt3_1,150,415). coords(fomdgt3_1,46,102,415).  visible(fomdgt3_1,415). happensAt(inactive(fomdgt3_1),415). orientation(fomdgt3_7,134,415). coords(fomdgt3_7,80,115,415).  visible(fomdgt3_7,415). happensAt(walking(fomdgt3_7),415).
orientation(fomdgt3_1,150,416). coords(fomdgt3_1,46,102,416).  visible(fomdgt3_1,416). happensAt(inactive(fomdgt3_1),416). orientation(fomdgt3_7,134,416). coords(fomdgt3_7,80,116,416).  visible(fomdgt3_7,416). happensAt(walking(fomdgt3_7),416).
orientation(fomdgt3_1,150,417). coords(fomdgt3_1,46,102,417).  visible(fomdgt3_1,417). happensAt(inactive(fomdgt3_1),417). orientation(fomdgt3_7,134,417). coords(fomdgt3_7,80,116,417).  visible(fomdgt3_7,417). happensAt(walking(fomdgt3_7),417).
orientation(fomdgt3_1,150,418). coords(fomdgt3_1,46,102,418).  visible(fomdgt3_1,418). happensAt(inactive(fomdgt3_1),418). orientation(fomdgt3_7,134,418). coords(fomdgt3_7,80,116,418).  visible(fomdgt3_7,418). happensAt(walking(fomdgt3_7),418).
orientation(fomdgt3_1,150,419). coords(fomdgt3_1,46,102,419).  visible(fomdgt3_1,419). happensAt(inactive(fomdgt3_1),419). orientation(fomdgt3_7,150,419). coords(fomdgt3_7,80,116,419).  visible(fomdgt3_7,419). happensAt(walking(fomdgt3_7),419).
orientation(fomdgt3_1,150,420). coords(fomdgt3_1,46,102,420).  visible(fomdgt3_1,420). happensAt(inactive(fomdgt3_1),420). orientation(fomdgt3_7,150,420). coords(fomdgt3_7,80,116,420).  visible(fomdgt3_7,420). happensAt(walking(fomdgt3_7),420).
orientation(fomdgt3_1,150,421). coords(fomdgt3_1,46,102,421).  visible(fomdgt3_1,421). happensAt(inactive(fomdgt3_1),421). orientation(fomdgt3_7,150,421). coords(fomdgt3_7,80,116,421).  visible(fomdgt3_7,421). happensAt(walking(fomdgt3_7),421).
orientation(fomdgt3_1,150,422). coords(fomdgt3_1,46,102,422).  visible(fomdgt3_1,422). happensAt(inactive(fomdgt3_1),422). orientation(fomdgt3_7,150,422). coords(fomdgt3_7,81,116,422).  visible(fomdgt3_7,422). happensAt(walking(fomdgt3_7),422).
#show holdsAt/2.