
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

	orientation(fomdgt3_1,150,402). coords(fomdgt3_1,46,102,402).  visible(fomdgt3_1,402). happensAt(inactive(fomdgt3_1),402). orientation(fomdgt3_6,140,402). coords(fomdgt3_6,79,81,402).  visible(fomdgt3_6,402). happensAt(running(fomdgt3_6),402).
orientation(fomdgt3_1,150,403). coords(fomdgt3_1,46,102,403).  visible(fomdgt3_1,403). happensAt(inactive(fomdgt3_1),403). orientation(fomdgt3_6,140,403). coords(fomdgt3_6,77,81,403).  visible(fomdgt3_6,403). happensAt(running(fomdgt3_6),403).
orientation(fomdgt3_1,150,404). coords(fomdgt3_1,46,102,404).  visible(fomdgt3_1,404). happensAt(inactive(fomdgt3_1),404). orientation(fomdgt3_6,140,404). coords(fomdgt3_6,76,81,404).  visible(fomdgt3_6,404). happensAt(running(fomdgt3_6),404).
orientation(fomdgt3_1,150,405). coords(fomdgt3_1,46,102,405).  visible(fomdgt3_1,405). happensAt(inactive(fomdgt3_1),405). orientation(fomdgt3_6,140,405). coords(fomdgt3_6,75,81,405).  visible(fomdgt3_6,405). happensAt(running(fomdgt3_6),405).
orientation(fomdgt3_1,150,406). coords(fomdgt3_1,46,102,406).  visible(fomdgt3_1,406). happensAt(inactive(fomdgt3_1),406). orientation(fomdgt3_6,140,406). coords(fomdgt3_6,74,81,406).  visible(fomdgt3_6,406). happensAt(running(fomdgt3_6),406).
orientation(fomdgt3_1,150,407). coords(fomdgt3_1,46,102,407).  visible(fomdgt3_1,407). happensAt(inactive(fomdgt3_1),407). orientation(fomdgt3_6,140,407). coords(fomdgt3_6,71,82,407).  visible(fomdgt3_6,407). happensAt(running(fomdgt3_6),407).
orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_6,140,408). coords(fomdgt3_6,71,82,408).  visible(fomdgt3_6,408). happensAt(running(fomdgt3_6),408).
orientation(fomdgt3_1,150,409). coords(fomdgt3_1,46,102,409).  visible(fomdgt3_1,409). happensAt(inactive(fomdgt3_1),409). orientation(fomdgt3_6,140,409). coords(fomdgt3_6,69,82,409).  visible(fomdgt3_6,409). happensAt(running(fomdgt3_6),409).
orientation(fomdgt3_1,150,410). coords(fomdgt3_1,46,102,410).  visible(fomdgt3_1,410). happensAt(inactive(fomdgt3_1),410). orientation(fomdgt3_6,140,410). coords(fomdgt3_6,68,82,410).  visible(fomdgt3_6,410). happensAt(running(fomdgt3_6),410).
orientation(fomdgt3_1,150,411). coords(fomdgt3_1,46,102,411).  visible(fomdgt3_1,411). happensAt(inactive(fomdgt3_1),411). orientation(fomdgt3_6,140,411). coords(fomdgt3_6,68,82,411).  visible(fomdgt3_6,411). happensAt(running(fomdgt3_6),411).
orientation(fomdgt3_1,150,412). coords(fomdgt3_1,46,102,412).  visible(fomdgt3_1,412). happensAt(inactive(fomdgt3_1),412). orientation(fomdgt3_6,140,412). coords(fomdgt3_6,67,83,412).  visible(fomdgt3_6,412). happensAt(running(fomdgt3_6),412).
orientation(fomdgt3_1,150,413). coords(fomdgt3_1,46,102,413).  visible(fomdgt3_1,413). happensAt(inactive(fomdgt3_1),413). orientation(fomdgt3_6,140,413). coords(fomdgt3_6,66,83,413).  visible(fomdgt3_6,413). happensAt(running(fomdgt3_6),413).
orientation(fomdgt3_1,150,414). coords(fomdgt3_1,46,102,414).  visible(fomdgt3_1,414). happensAt(inactive(fomdgt3_1),414). orientation(fomdgt3_6,140,414). coords(fomdgt3_6,64,83,414).  visible(fomdgt3_6,414). happensAt(running(fomdgt3_6),414).
orientation(fomdgt3_1,150,415). coords(fomdgt3_1,46,102,415).  visible(fomdgt3_1,415). happensAt(inactive(fomdgt3_1),415). orientation(fomdgt3_6,140,415). coords(fomdgt3_6,63,84,415).  visible(fomdgt3_6,415). happensAt(running(fomdgt3_6),415).
orientation(fomdgt3_1,150,416). coords(fomdgt3_1,46,102,416).  visible(fomdgt3_1,416). happensAt(inactive(fomdgt3_1),416). orientation(fomdgt3_6,140,416). coords(fomdgt3_6,62,84,416).  visible(fomdgt3_6,416). happensAt(running(fomdgt3_6),416).
orientation(fomdgt3_1,150,417). coords(fomdgt3_1,46,102,417).  visible(fomdgt3_1,417). happensAt(inactive(fomdgt3_1),417). orientation(fomdgt3_6,140,417). coords(fomdgt3_6,62,84,417).  visible(fomdgt3_6,417). happensAt(running(fomdgt3_6),417).
orientation(fomdgt3_1,150,418). coords(fomdgt3_1,46,102,418).  visible(fomdgt3_1,418). happensAt(inactive(fomdgt3_1),418). orientation(fomdgt3_6,140,418). coords(fomdgt3_6,60,85,418).  visible(fomdgt3_6,418). happensAt(running(fomdgt3_6),418).
orientation(fomdgt3_1,150,419). coords(fomdgt3_1,46,102,419).  visible(fomdgt3_1,419). happensAt(inactive(fomdgt3_1),419). orientation(fomdgt3_6,140,419). coords(fomdgt3_6,60,88,419).  visible(fomdgt3_6,419). happensAt(running(fomdgt3_6),419).
orientation(fomdgt3_1,150,420). coords(fomdgt3_1,46,102,420).  visible(fomdgt3_1,420). happensAt(inactive(fomdgt3_1),420). orientation(fomdgt3_6,140,420). coords(fomdgt3_6,60,88,420).  visible(fomdgt3_6,420). happensAt(running(fomdgt3_6),420).
orientation(fomdgt3_1,150,421). coords(fomdgt3_1,46,102,421).  visible(fomdgt3_1,421). happensAt(inactive(fomdgt3_1),421). orientation(fomdgt3_6,140,421). coords(fomdgt3_6,60,89,421).  visible(fomdgt3_6,421). happensAt(walking(fomdgt3_6),421).
orientation(fomdgt3_1,150,422). coords(fomdgt3_1,46,102,422).  visible(fomdgt3_1,422). happensAt(inactive(fomdgt3_1),422). orientation(fomdgt3_6,140,422). coords(fomdgt3_6,59,89,422).  visible(fomdgt3_6,422). happensAt(walking(fomdgt3_6),422).
orientation(fomdgt3_1,150,423). coords(fomdgt3_1,46,102,423).  visible(fomdgt3_1,423). happensAt(inactive(fomdgt3_1),423). orientation(fomdgt3_6,140,423). coords(fomdgt3_6,58,89,423).  visible(fomdgt3_6,423). happensAt(walking(fomdgt3_6),423).
orientation(fomdgt3_1,150,424). coords(fomdgt3_1,46,102,424).  visible(fomdgt3_1,424). happensAt(inactive(fomdgt3_1),424). orientation(fomdgt3_6,160,424). coords(fomdgt3_6,54,90,424).  visible(fomdgt3_6,424). happensAt(walking(fomdgt3_6),424).
orientation(fomdgt3_1,150,425). coords(fomdgt3_1,46,102,425).  visible(fomdgt3_1,425). happensAt(inactive(fomdgt3_1),425). orientation(fomdgt3_6,160,425). coords(fomdgt3_6,53,91,425).  visible(fomdgt3_6,425). happensAt(walking(fomdgt3_6),425).
orientation(fomdgt3_1,150,426). coords(fomdgt3_1,46,102,426).  visible(fomdgt3_1,426). happensAt(inactive(fomdgt3_1),426). orientation(fomdgt3_6,160,426). coords(fomdgt3_6,53,91,426).  visible(fomdgt3_6,426). happensAt(walking(fomdgt3_6),426).
#show holdsAt/2.