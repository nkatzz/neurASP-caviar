
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

	orientation(fcgt_1,128,402). coords(fcgt_1,72,84,402).  visible(fcgt_1,402). happensAt(running(fcgt_1),402). orientation(fcgt_4,166,402). coords(fcgt_4,82,206,402).  visible(fcgt_4,402). happensAt(inactive(fcgt_4),402).
orientation(fcgt_1,128,403). coords(fcgt_1,72,82,403).  visible(fcgt_1,403). happensAt(running(fcgt_1),403). orientation(fcgt_4,166,403). coords(fcgt_4,82,206,403).  visible(fcgt_4,403). happensAt(inactive(fcgt_4),403).
orientation(fcgt_1,128,404). coords(fcgt_1,72,80,404).  visible(fcgt_1,404). happensAt(running(fcgt_1),404). orientation(fcgt_4,166,404). coords(fcgt_4,82,206,404).  visible(fcgt_4,404). happensAt(inactive(fcgt_4),404).
orientation(fcgt_1,128,405). coords(fcgt_1,72,78,405).  visible(fcgt_1,405). happensAt(running(fcgt_1),405). orientation(fcgt_4,166,405). coords(fcgt_4,82,206,405).  visible(fcgt_4,405). happensAt(inactive(fcgt_4),405).
orientation(fcgt_1,128,406). coords(fcgt_1,72,77,406).  visible(fcgt_1,406). happensAt(running(fcgt_1),406). orientation(fcgt_4,166,406). coords(fcgt_4,82,206,406).  visible(fcgt_4,406). happensAt(inactive(fcgt_4),406).
orientation(fcgt_1,128,407). coords(fcgt_1,72,77,407).  visible(fcgt_1,407). happensAt(running(fcgt_1),407). orientation(fcgt_4,166,407). coords(fcgt_4,82,206,407).  visible(fcgt_4,407). happensAt(inactive(fcgt_4),407).
orientation(fcgt_1,128,408). coords(fcgt_1,72,75,408).  visible(fcgt_1,408). happensAt(running(fcgt_1),408). orientation(fcgt_4,166,408). coords(fcgt_4,82,206,408).  visible(fcgt_4,408). happensAt(inactive(fcgt_4),408).
orientation(fcgt_1,128,409). coords(fcgt_1,72,75,409).  visible(fcgt_1,409). happensAt(running(fcgt_1),409). orientation(fcgt_4,166,409). coords(fcgt_4,82,206,409).  visible(fcgt_4,409). happensAt(inactive(fcgt_4),409).
orientation(fcgt_1,128,410). coords(fcgt_1,73,74,410).  visible(fcgt_1,410). happensAt(running(fcgt_1),410). orientation(fcgt_4,166,410). coords(fcgt_4,82,206,410).  visible(fcgt_4,410). happensAt(inactive(fcgt_4),410).
orientation(fcgt_1,128,411). coords(fcgt_1,73,73,411).  visible(fcgt_1,411). happensAt(running(fcgt_1),411). orientation(fcgt_4,166,411). coords(fcgt_4,82,206,411).  visible(fcgt_4,411). happensAt(inactive(fcgt_4),411).
orientation(fcgt_1,128,412). coords(fcgt_1,73,71,412).  visible(fcgt_1,412). happensAt(running(fcgt_1),412). orientation(fcgt_4,166,412). coords(fcgt_4,82,206,412).  visible(fcgt_4,412). happensAt(inactive(fcgt_4),412).
orientation(fcgt_1,128,413). coords(fcgt_1,73,70,413).  visible(fcgt_1,413). happensAt(running(fcgt_1),413). orientation(fcgt_4,166,413). coords(fcgt_4,82,206,413).  visible(fcgt_4,413). happensAt(inactive(fcgt_4),413).
orientation(fcgt_1,128,414). coords(fcgt_1,73,69,414).  visible(fcgt_1,414). happensAt(running(fcgt_1),414). orientation(fcgt_4,166,414). coords(fcgt_4,82,206,414).  visible(fcgt_4,414). happensAt(inactive(fcgt_4),414).
orientation(fcgt_1,128,415). coords(fcgt_1,73,68,415).  visible(fcgt_1,415). happensAt(running(fcgt_1),415). orientation(fcgt_4,166,415). coords(fcgt_4,82,206,415).  visible(fcgt_4,415). happensAt(inactive(fcgt_4),415).
orientation(fcgt_1,128,416). coords(fcgt_1,74,68,416).  visible(fcgt_1,416). happensAt(running(fcgt_1),416). orientation(fcgt_4,166,416). coords(fcgt_4,82,206,416).  visible(fcgt_4,416). happensAt(inactive(fcgt_4),416).
orientation(fcgt_1,128,417). coords(fcgt_1,74,67,417).  visible(fcgt_1,417). happensAt(running(fcgt_1),417). orientation(fcgt_4,166,417). coords(fcgt_4,82,206,417).  visible(fcgt_4,417). happensAt(inactive(fcgt_4),417).
orientation(fcgt_1,128,418). coords(fcgt_1,74,66,418).  visible(fcgt_1,418). happensAt(running(fcgt_1),418). orientation(fcgt_4,166,418). coords(fcgt_4,82,206,418).  visible(fcgt_4,418). happensAt(inactive(fcgt_4),418).
orientation(fcgt_1,128,419). coords(fcgt_1,72,65,419).  visible(fcgt_1,419). happensAt(running(fcgt_1),419). orientation(fcgt_4,166,419). coords(fcgt_4,82,206,419).  visible(fcgt_4,419). happensAt(inactive(fcgt_4),419).
orientation(fcgt_1,128,420). coords(fcgt_1,76,63,420).  visible(fcgt_1,420). happensAt(running(fcgt_1),420). orientation(fcgt_4,166,420). coords(fcgt_4,82,206,420).  visible(fcgt_4,420). happensAt(inactive(fcgt_4),420).
orientation(fcgt_1,128,421). coords(fcgt_1,78,61,421).  visible(fcgt_1,421). happensAt(running(fcgt_1),421). orientation(fcgt_4,166,421). coords(fcgt_4,82,206,421).  visible(fcgt_4,421). happensAt(inactive(fcgt_4),421).
orientation(fcgt_1,128,422). coords(fcgt_1,77,62,422).  visible(fcgt_1,422). happensAt(running(fcgt_1),422). orientation(fcgt_4,166,422). coords(fcgt_4,82,206,422).  visible(fcgt_4,422). happensAt(inactive(fcgt_4),422).
orientation(fcgt_1,128,423). coords(fcgt_1,77,60,423).  visible(fcgt_1,423). happensAt(running(fcgt_1),423). orientation(fcgt_4,166,423). coords(fcgt_4,82,206,423).  visible(fcgt_4,423). happensAt(inactive(fcgt_4),423).
orientation(fcgt_1,128,424). coords(fcgt_1,77,60,424).  visible(fcgt_1,424). happensAt(running(fcgt_1),424). orientation(fcgt_4,166,424). coords(fcgt_4,82,206,424).  visible(fcgt_4,424). happensAt(inactive(fcgt_4),424).
orientation(fcgt_1,128,425). coords(fcgt_1,77,60,425).  visible(fcgt_1,425). happensAt(running(fcgt_1),425). orientation(fcgt_4,166,425). coords(fcgt_4,82,206,425).  visible(fcgt_4,425). happensAt(inactive(fcgt_4),425).
orientation(fcgt_1,128,426). coords(fcgt_1,77,60,426).  happensAt(disappear(fcgt_1),426). happensAt(running(fcgt_1),426). orientation(fcgt_4,166,426). coords(fcgt_4,82,206,426).  visible(fcgt_4,426). happensAt(inactive(fcgt_4),426).
#show holdsAt/2.