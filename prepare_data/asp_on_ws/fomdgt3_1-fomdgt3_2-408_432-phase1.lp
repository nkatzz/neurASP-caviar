
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

	orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_2,42,408). coords(fomdgt3_2,262,96,408).  visible(fomdgt3_2,408). happensAt(inactive(fomdgt3_2),408).
orientation(fomdgt3_1,150,409). coords(fomdgt3_1,46,102,409).  visible(fomdgt3_1,409). happensAt(inactive(fomdgt3_1),409). orientation(fomdgt3_2,42,409). coords(fomdgt3_2,262,96,409).  visible(fomdgt3_2,409). happensAt(inactive(fomdgt3_2),409).
orientation(fomdgt3_1,150,410). coords(fomdgt3_1,46,102,410).  visible(fomdgt3_1,410). happensAt(inactive(fomdgt3_1),410). orientation(fomdgt3_2,42,410). coords(fomdgt3_2,262,96,410).  visible(fomdgt3_2,410). happensAt(inactive(fomdgt3_2),410).
orientation(fomdgt3_1,150,411). coords(fomdgt3_1,46,102,411).  visible(fomdgt3_1,411). happensAt(inactive(fomdgt3_1),411). orientation(fomdgt3_2,42,411). coords(fomdgt3_2,262,96,411).  visible(fomdgt3_2,411). happensAt(inactive(fomdgt3_2),411).
orientation(fomdgt3_1,150,412). coords(fomdgt3_1,46,102,412).  visible(fomdgt3_1,412). happensAt(inactive(fomdgt3_1),412). orientation(fomdgt3_2,42,412). coords(fomdgt3_2,262,96,412).  visible(fomdgt3_2,412). happensAt(inactive(fomdgt3_2),412).
orientation(fomdgt3_1,150,413). coords(fomdgt3_1,46,102,413).  visible(fomdgt3_1,413). happensAt(inactive(fomdgt3_1),413). orientation(fomdgt3_2,42,413). coords(fomdgt3_2,262,96,413).  visible(fomdgt3_2,413). happensAt(inactive(fomdgt3_2),413).
orientation(fomdgt3_1,150,414). coords(fomdgt3_1,46,102,414).  visible(fomdgt3_1,414). happensAt(inactive(fomdgt3_1),414). orientation(fomdgt3_2,42,414). coords(fomdgt3_2,262,96,414).  visible(fomdgt3_2,414). happensAt(inactive(fomdgt3_2),414).
orientation(fomdgt3_1,150,415). coords(fomdgt3_1,46,102,415).  visible(fomdgt3_1,415). happensAt(inactive(fomdgt3_1),415). orientation(fomdgt3_2,42,415). coords(fomdgt3_2,262,96,415).  visible(fomdgt3_2,415). happensAt(inactive(fomdgt3_2),415).
orientation(fomdgt3_1,150,416). coords(fomdgt3_1,46,102,416).  visible(fomdgt3_1,416). happensAt(inactive(fomdgt3_1),416). orientation(fomdgt3_2,42,416). coords(fomdgt3_2,262,96,416).  visible(fomdgt3_2,416). happensAt(inactive(fomdgt3_2),416).
orientation(fomdgt3_1,150,417). coords(fomdgt3_1,46,102,417).  visible(fomdgt3_1,417). happensAt(inactive(fomdgt3_1),417). orientation(fomdgt3_2,42,417). coords(fomdgt3_2,262,96,417).  visible(fomdgt3_2,417). happensAt(inactive(fomdgt3_2),417).
orientation(fomdgt3_1,150,418). coords(fomdgt3_1,46,102,418).  visible(fomdgt3_1,418). happensAt(inactive(fomdgt3_1),418). orientation(fomdgt3_2,42,418). coords(fomdgt3_2,262,96,418).  visible(fomdgt3_2,418). happensAt(inactive(fomdgt3_2),418).
orientation(fomdgt3_1,150,419). coords(fomdgt3_1,46,102,419).  visible(fomdgt3_1,419). happensAt(inactive(fomdgt3_1),419). orientation(fomdgt3_2,42,419). coords(fomdgt3_2,262,96,419).  visible(fomdgt3_2,419). happensAt(inactive(fomdgt3_2),419).
orientation(fomdgt3_1,150,420). coords(fomdgt3_1,46,102,420).  visible(fomdgt3_1,420). happensAt(inactive(fomdgt3_1),420). orientation(fomdgt3_2,42,420). coords(fomdgt3_2,262,96,420).  visible(fomdgt3_2,420). happensAt(inactive(fomdgt3_2),420).
orientation(fomdgt3_1,150,421). coords(fomdgt3_1,46,102,421).  visible(fomdgt3_1,421). happensAt(inactive(fomdgt3_1),421). orientation(fomdgt3_2,42,421). coords(fomdgt3_2,262,96,421).  visible(fomdgt3_2,421). happensAt(inactive(fomdgt3_2),421).
orientation(fomdgt3_1,150,422). coords(fomdgt3_1,46,102,422).  visible(fomdgt3_1,422). happensAt(inactive(fomdgt3_1),422). orientation(fomdgt3_2,42,422). coords(fomdgt3_2,262,96,422).  visible(fomdgt3_2,422). happensAt(inactive(fomdgt3_2),422).
orientation(fomdgt3_1,150,423). coords(fomdgt3_1,46,102,423).  visible(fomdgt3_1,423). happensAt(inactive(fomdgt3_1),423). orientation(fomdgt3_2,42,423). coords(fomdgt3_2,262,96,423).  visible(fomdgt3_2,423). happensAt(inactive(fomdgt3_2),423).
orientation(fomdgt3_1,150,424). coords(fomdgt3_1,46,102,424).  visible(fomdgt3_1,424). happensAt(inactive(fomdgt3_1),424). orientation(fomdgt3_2,42,424). coords(fomdgt3_2,262,96,424).  visible(fomdgt3_2,424). happensAt(inactive(fomdgt3_2),424).
orientation(fomdgt3_1,150,425). coords(fomdgt3_1,46,102,425).  visible(fomdgt3_1,425). happensAt(inactive(fomdgt3_1),425). orientation(fomdgt3_2,42,425). coords(fomdgt3_2,262,96,425).  visible(fomdgt3_2,425). happensAt(inactive(fomdgt3_2),425).
orientation(fomdgt3_1,150,426). coords(fomdgt3_1,46,102,426).  visible(fomdgt3_1,426). happensAt(inactive(fomdgt3_1),426). orientation(fomdgt3_2,42,426). coords(fomdgt3_2,262,96,426).  visible(fomdgt3_2,426). happensAt(inactive(fomdgt3_2),426).
orientation(fomdgt3_1,150,427). coords(fomdgt3_1,46,102,427).  visible(fomdgt3_1,427). happensAt(inactive(fomdgt3_1),427). orientation(fomdgt3_2,42,427). coords(fomdgt3_2,262,96,427).  visible(fomdgt3_2,427). happensAt(inactive(fomdgt3_2),427).
orientation(fomdgt3_1,150,428). coords(fomdgt3_1,46,102,428).  visible(fomdgt3_1,428). happensAt(inactive(fomdgt3_1),428). orientation(fomdgt3_2,42,428). coords(fomdgt3_2,262,96,428).  visible(fomdgt3_2,428). happensAt(inactive(fomdgt3_2),428).
orientation(fomdgt3_1,150,429). coords(fomdgt3_1,46,102,429).  visible(fomdgt3_1,429). happensAt(inactive(fomdgt3_1),429). orientation(fomdgt3_2,42,429). coords(fomdgt3_2,262,96,429).  visible(fomdgt3_2,429). happensAt(inactive(fomdgt3_2),429).
orientation(fomdgt3_1,150,430). coords(fomdgt3_1,46,102,430).  visible(fomdgt3_1,430). happensAt(inactive(fomdgt3_1),430). orientation(fomdgt3_2,42,430). coords(fomdgt3_2,262,96,430).  visible(fomdgt3_2,430). happensAt(inactive(fomdgt3_2),430).
orientation(fomdgt3_1,150,431). coords(fomdgt3_1,46,102,431).  visible(fomdgt3_1,431). happensAt(inactive(fomdgt3_1),431). orientation(fomdgt3_2,42,431). coords(fomdgt3_2,262,96,431).  visible(fomdgt3_2,431). happensAt(inactive(fomdgt3_2),431).
orientation(fomdgt3_1,150,432). coords(fomdgt3_1,46,102,432).  visible(fomdgt3_1,432). happensAt(inactive(fomdgt3_1),432). orientation(fomdgt3_2,42,432). coords(fomdgt3_2,262,96,432).  visible(fomdgt3_2,432). happensAt(inactive(fomdgt3_2),432).
#show holdsAt/2.