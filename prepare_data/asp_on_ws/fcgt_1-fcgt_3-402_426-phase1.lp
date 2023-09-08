
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

	orientation(fcgt_1,128,402). coords(fcgt_1,72,84,402).  visible(fcgt_1,402). happensAt(running(fcgt_1),402). orientation(fcgt_3,164,402). coords(fcgt_3,99,197,402).  visible(fcgt_3,402). happensAt(walking(fcgt_3),402).
orientation(fcgt_1,128,403). coords(fcgt_1,72,82,403).  visible(fcgt_1,403). happensAt(running(fcgt_1),403). orientation(fcgt_3,164,403). coords(fcgt_3,99,198,403).  visible(fcgt_3,403). happensAt(walking(fcgt_3),403).
orientation(fcgt_1,128,404). coords(fcgt_1,72,80,404).  visible(fcgt_1,404). happensAt(running(fcgt_1),404). orientation(fcgt_3,164,404). coords(fcgt_3,99,198,404).  visible(fcgt_3,404). happensAt(walking(fcgt_3),404).
orientation(fcgt_1,128,405). coords(fcgt_1,72,78,405).  visible(fcgt_1,405). happensAt(running(fcgt_1),405). orientation(fcgt_3,164,405). coords(fcgt_3,99,198,405).  visible(fcgt_3,405). happensAt(walking(fcgt_3),405).
orientation(fcgt_1,128,406). coords(fcgt_1,72,77,406).  visible(fcgt_1,406). happensAt(running(fcgt_1),406). orientation(fcgt_3,164,406). coords(fcgt_3,99,199,406).  visible(fcgt_3,406). happensAt(walking(fcgt_3),406).
orientation(fcgt_1,128,407). coords(fcgt_1,72,77,407).  visible(fcgt_1,407). happensAt(running(fcgt_1),407). orientation(fcgt_3,164,407). coords(fcgt_3,99,199,407).  visible(fcgt_3,407). happensAt(walking(fcgt_3),407).
orientation(fcgt_1,128,408). coords(fcgt_1,72,75,408).  visible(fcgt_1,408). happensAt(running(fcgt_1),408). orientation(fcgt_3,164,408). coords(fcgt_3,99,200,408).  visible(fcgt_3,408). happensAt(walking(fcgt_3),408).
orientation(fcgt_1,128,409). coords(fcgt_1,72,75,409).  visible(fcgt_1,409). happensAt(running(fcgt_1),409). orientation(fcgt_3,164,409). coords(fcgt_3,99,200,409).  visible(fcgt_3,409). happensAt(walking(fcgt_3),409).
orientation(fcgt_1,128,410). coords(fcgt_1,73,74,410).  visible(fcgt_1,410). happensAt(running(fcgt_1),410). orientation(fcgt_3,164,410). coords(fcgt_3,99,200,410).  visible(fcgt_3,410). happensAt(walking(fcgt_3),410).
orientation(fcgt_1,128,411). coords(fcgt_1,73,73,411).  visible(fcgt_1,411). happensAt(running(fcgt_1),411). orientation(fcgt_3,164,411). coords(fcgt_3,99,200,411).  visible(fcgt_3,411). happensAt(walking(fcgt_3),411).
orientation(fcgt_1,128,412). coords(fcgt_1,73,71,412).  visible(fcgt_1,412). happensAt(running(fcgt_1),412). orientation(fcgt_3,164,412). coords(fcgt_3,98,200,412).  visible(fcgt_3,412). happensAt(walking(fcgt_3),412).
orientation(fcgt_1,128,413). coords(fcgt_1,73,70,413).  visible(fcgt_1,413). happensAt(running(fcgt_1),413). orientation(fcgt_3,164,413). coords(fcgt_3,98,200,413).  visible(fcgt_3,413). happensAt(walking(fcgt_3),413).
orientation(fcgt_1,128,414). coords(fcgt_1,73,69,414).  visible(fcgt_1,414). happensAt(running(fcgt_1),414). orientation(fcgt_3,164,414). coords(fcgt_3,98,200,414).  visible(fcgt_3,414). happensAt(walking(fcgt_3),414).
orientation(fcgt_1,128,415). coords(fcgt_1,73,68,415).  visible(fcgt_1,415). happensAt(running(fcgt_1),415). orientation(fcgt_3,164,415). coords(fcgt_3,98,201,415).  visible(fcgt_3,415). happensAt(walking(fcgt_3),415).
orientation(fcgt_1,128,416). coords(fcgt_1,74,68,416).  visible(fcgt_1,416). happensAt(running(fcgt_1),416). orientation(fcgt_3,164,416). coords(fcgt_3,98,201,416).  visible(fcgt_3,416). happensAt(walking(fcgt_3),416).
orientation(fcgt_1,128,417). coords(fcgt_1,74,67,417).  visible(fcgt_1,417). happensAt(running(fcgt_1),417). orientation(fcgt_3,164,417). coords(fcgt_3,98,201,417).  visible(fcgt_3,417). happensAt(walking(fcgt_3),417).
orientation(fcgt_1,128,418). coords(fcgt_1,74,66,418).  visible(fcgt_1,418). happensAt(running(fcgt_1),418). orientation(fcgt_3,164,418). coords(fcgt_3,98,201,418).  visible(fcgt_3,418). happensAt(walking(fcgt_3),418).
orientation(fcgt_1,128,419). coords(fcgt_1,72,65,419).  visible(fcgt_1,419). happensAt(running(fcgt_1),419). orientation(fcgt_3,164,419). coords(fcgt_3,98,201,419).  visible(fcgt_3,419). happensAt(walking(fcgt_3),419).
orientation(fcgt_1,128,420). coords(fcgt_1,76,63,420).  visible(fcgt_1,420). happensAt(running(fcgt_1),420). orientation(fcgt_3,164,420). coords(fcgt_3,98,201,420).  visible(fcgt_3,420). happensAt(walking(fcgt_3),420).
orientation(fcgt_1,128,421). coords(fcgt_1,78,61,421).  visible(fcgt_1,421). happensAt(running(fcgt_1),421). orientation(fcgt_3,164,421). coords(fcgt_3,98,201,421).  visible(fcgt_3,421). happensAt(walking(fcgt_3),421).
orientation(fcgt_1,128,422). coords(fcgt_1,77,62,422).  visible(fcgt_1,422). happensAt(running(fcgt_1),422). orientation(fcgt_3,164,422). coords(fcgt_3,98,201,422).  visible(fcgt_3,422). happensAt(walking(fcgt_3),422).
orientation(fcgt_1,128,423). coords(fcgt_1,77,60,423).  visible(fcgt_1,423). happensAt(running(fcgt_1),423). orientation(fcgt_3,164,423). coords(fcgt_3,98,201,423).  visible(fcgt_3,423). happensAt(walking(fcgt_3),423).
orientation(fcgt_1,128,424). coords(fcgt_1,77,60,424).  visible(fcgt_1,424). happensAt(running(fcgt_1),424). orientation(fcgt_3,164,424). coords(fcgt_3,98,201,424).  visible(fcgt_3,424). happensAt(walking(fcgt_3),424).
orientation(fcgt_1,128,425). coords(fcgt_1,77,60,425).  visible(fcgt_1,425). happensAt(running(fcgt_1),425). orientation(fcgt_3,164,425). coords(fcgt_3,98,201,425).  visible(fcgt_3,425). happensAt(walking(fcgt_3),425).
orientation(fcgt_1,128,426). coords(fcgt_1,77,60,426).  happensAt(disappear(fcgt_1),426). happensAt(running(fcgt_1),426). orientation(fcgt_3,164,426). coords(fcgt_3,98,201,426).  visible(fcgt_3,426). happensAt(walking(fcgt_3),426).
#show holdsAt/2.