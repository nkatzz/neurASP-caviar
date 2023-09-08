
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

	orientation(mws1gt_3,165,399). coords(mws1gt_3,37,161,399).  visible(mws1gt_3,399). happensAt(walking(mws1gt_3),399). orientation(mws1gt_7,174,399). coords(mws1gt_7,62,188,399).  visible(mws1gt_7,399). happensAt(walking(mws1gt_7),399).
orientation(mws1gt_3,165,400). coords(mws1gt_3,37,161,400).  visible(mws1gt_3,400). happensAt(walking(mws1gt_3),400). orientation(mws1gt_7,174,400). coords(mws1gt_7,63,188,400).  visible(mws1gt_7,400). happensAt(walking(mws1gt_7),400).
orientation(mws1gt_3,165,401). coords(mws1gt_3,36,161,401).  visible(mws1gt_3,401). happensAt(walking(mws1gt_3),401). orientation(mws1gt_7,174,401). coords(mws1gt_7,63,188,401).  visible(mws1gt_7,401). happensAt(walking(mws1gt_7),401).
orientation(mws1gt_3,165,402). coords(mws1gt_3,36,161,402).  visible(mws1gt_3,402). happensAt(walking(mws1gt_3),402). orientation(mws1gt_7,174,402). coords(mws1gt_7,65,188,402).  visible(mws1gt_7,402). happensAt(walking(mws1gt_7),402).
orientation(mws1gt_3,165,403). coords(mws1gt_3,36,161,403).  visible(mws1gt_3,403). happensAt(walking(mws1gt_3),403). orientation(mws1gt_7,174,403). coords(mws1gt_7,66,189,403).  visible(mws1gt_7,403). happensAt(walking(mws1gt_7),403).
orientation(mws1gt_3,165,404). coords(mws1gt_3,36,161,404).  visible(mws1gt_3,404). happensAt(walking(mws1gt_3),404). orientation(mws1gt_7,174,404). coords(mws1gt_7,66,189,404).  visible(mws1gt_7,404). happensAt(walking(mws1gt_7),404).
orientation(mws1gt_3,165,405). coords(mws1gt_3,36,161,405).  visible(mws1gt_3,405). happensAt(walking(mws1gt_3),405). orientation(mws1gt_7,174,405). coords(mws1gt_7,66,189,405).  visible(mws1gt_7,405). happensAt(walking(mws1gt_7),405).
orientation(mws1gt_3,165,406). coords(mws1gt_3,36,161,406).  visible(mws1gt_3,406). happensAt(walking(mws1gt_3),406). orientation(mws1gt_7,174,406). coords(mws1gt_7,66,189,406).  visible(mws1gt_7,406). happensAt(walking(mws1gt_7),406).
orientation(mws1gt_3,165,407). coords(mws1gt_3,35,161,407).  visible(mws1gt_3,407). happensAt(walking(mws1gt_3),407). orientation(mws1gt_7,174,407). coords(mws1gt_7,67,190,407).  visible(mws1gt_7,407). happensAt(walking(mws1gt_7),407).
orientation(mws1gt_3,165,408). coords(mws1gt_3,35,161,408).  visible(mws1gt_3,408). happensAt(walking(mws1gt_3),408). orientation(mws1gt_7,174,408). coords(mws1gt_7,67,190,408).  visible(mws1gt_7,408). happensAt(walking(mws1gt_7),408).
orientation(mws1gt_3,165,409). coords(mws1gt_3,34,160,409).  visible(mws1gt_3,409). happensAt(walking(mws1gt_3),409). orientation(mws1gt_7,174,409). coords(mws1gt_7,67,190,409).  visible(mws1gt_7,409). happensAt(walking(mws1gt_7),409).
orientation(mws1gt_3,165,410). coords(mws1gt_3,34,160,410).  visible(mws1gt_3,410). happensAt(walking(mws1gt_3),410). orientation(mws1gt_7,174,410). coords(mws1gt_7,67,190,410).  visible(mws1gt_7,410). happensAt(walking(mws1gt_7),410).
orientation(mws1gt_3,165,411). coords(mws1gt_3,34,160,411).  visible(mws1gt_3,411). happensAt(walking(mws1gt_3),411). orientation(mws1gt_7,174,411). coords(mws1gt_7,67,190,411).  visible(mws1gt_7,411). happensAt(walking(mws1gt_7),411).
orientation(mws1gt_3,165,412). coords(mws1gt_3,33,160,412).  visible(mws1gt_3,412). happensAt(walking(mws1gt_3),412). orientation(mws1gt_7,174,412). coords(mws1gt_7,67,190,412).  visible(mws1gt_7,412). happensAt(walking(mws1gt_7),412).
orientation(mws1gt_3,165,413). coords(mws1gt_3,33,160,413).  visible(mws1gt_3,413). happensAt(walking(mws1gt_3),413). orientation(mws1gt_7,174,413). coords(mws1gt_7,67,190,413).  visible(mws1gt_7,413). happensAt(walking(mws1gt_7),413).
orientation(mws1gt_3,165,414). coords(mws1gt_3,33,160,414).  visible(mws1gt_3,414). happensAt(walking(mws1gt_3),414). orientation(mws1gt_7,174,414). coords(mws1gt_7,68,191,414).  visible(mws1gt_7,414). happensAt(walking(mws1gt_7),414).
orientation(mws1gt_3,165,415). coords(mws1gt_3,33,160,415).  visible(mws1gt_3,415). happensAt(walking(mws1gt_3),415). orientation(mws1gt_7,174,415). coords(mws1gt_7,69,191,415).  visible(mws1gt_7,415). happensAt(walking(mws1gt_7),415).
orientation(mws1gt_3,165,416). coords(mws1gt_3,33,160,416).  visible(mws1gt_3,416). happensAt(walking(mws1gt_3),416). orientation(mws1gt_7,174,416). coords(mws1gt_7,70,192,416).  visible(mws1gt_7,416). happensAt(walking(mws1gt_7),416).
orientation(mws1gt_3,165,417). coords(mws1gt_3,33,160,417).  visible(mws1gt_3,417). happensAt(walking(mws1gt_3),417). orientation(mws1gt_7,171,417). coords(mws1gt_7,70,192,417).  visible(mws1gt_7,417). happensAt(walking(mws1gt_7),417).
orientation(mws1gt_3,165,418). coords(mws1gt_3,33,160,418).  visible(mws1gt_3,418). happensAt(walking(mws1gt_3),418). orientation(mws1gt_7,168,418). coords(mws1gt_7,71,192,418).  visible(mws1gt_7,418). happensAt(walking(mws1gt_7),418).
orientation(mws1gt_3,165,419). coords(mws1gt_3,33,160,419).  visible(mws1gt_3,419). happensAt(walking(mws1gt_3),419). orientation(mws1gt_7,169,419). coords(mws1gt_7,71,192,419).  visible(mws1gt_7,419). happensAt(walking(mws1gt_7),419).
orientation(mws1gt_3,165,420). coords(mws1gt_3,32,160,420).  visible(mws1gt_3,420). happensAt(walking(mws1gt_3),420). orientation(mws1gt_7,168,420). coords(mws1gt_7,72,192,420).  visible(mws1gt_7,420). happensAt(walking(mws1gt_7),420).
orientation(mws1gt_3,165,421). coords(mws1gt_3,32,160,421).  visible(mws1gt_3,421). happensAt(walking(mws1gt_3),421). orientation(mws1gt_7,168,421). coords(mws1gt_7,73,193,421).  visible(mws1gt_7,421). happensAt(walking(mws1gt_7),421).
orientation(mws1gt_3,165,422). coords(mws1gt_3,32,160,422).  visible(mws1gt_3,422). happensAt(walking(mws1gt_3),422). orientation(mws1gt_7,163,422). coords(mws1gt_7,73,193,422).  visible(mws1gt_7,422). happensAt(walking(mws1gt_7),422).
orientation(mws1gt_3,165,423). coords(mws1gt_3,32,160,423).  visible(mws1gt_3,423). happensAt(walking(mws1gt_3),423). orientation(mws1gt_7,163,423). coords(mws1gt_7,73,193,423).  visible(mws1gt_7,423). happensAt(walking(mws1gt_7),423).
#show holdsAt/2.