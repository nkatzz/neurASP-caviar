
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

	orientation(fomdgt3_2,42,410). coords(fomdgt3_2,262,96,410).  visible(fomdgt3_2,410). happensAt(inactive(fomdgt3_2),410). orientation(fomdgt3_7,134,410). coords(fomdgt3_7,80,109,410).  visible(fomdgt3_7,410). happensAt(walking(fomdgt3_7),410).
orientation(fomdgt3_2,42,411). coords(fomdgt3_2,262,96,411).  visible(fomdgt3_2,411). happensAt(inactive(fomdgt3_2),411). orientation(fomdgt3_7,134,411). coords(fomdgt3_7,80,110,411).  visible(fomdgt3_7,411). happensAt(walking(fomdgt3_7),411).
orientation(fomdgt3_2,42,412). coords(fomdgt3_2,262,96,412).  visible(fomdgt3_2,412). happensAt(inactive(fomdgt3_2),412). orientation(fomdgt3_7,134,412). coords(fomdgt3_7,80,111,412).  visible(fomdgt3_7,412). happensAt(walking(fomdgt3_7),412).
orientation(fomdgt3_2,42,413). coords(fomdgt3_2,262,96,413).  visible(fomdgt3_2,413). happensAt(inactive(fomdgt3_2),413). orientation(fomdgt3_7,134,413). coords(fomdgt3_7,80,112,413).  visible(fomdgt3_7,413). happensAt(walking(fomdgt3_7),413).
orientation(fomdgt3_2,42,414). coords(fomdgt3_2,262,96,414).  visible(fomdgt3_2,414). happensAt(inactive(fomdgt3_2),414). orientation(fomdgt3_7,134,414). coords(fomdgt3_7,80,113,414).  visible(fomdgt3_7,414). happensAt(walking(fomdgt3_7),414).
orientation(fomdgt3_2,42,415). coords(fomdgt3_2,262,96,415).  visible(fomdgt3_2,415). happensAt(inactive(fomdgt3_2),415). orientation(fomdgt3_7,134,415). coords(fomdgt3_7,80,115,415).  visible(fomdgt3_7,415). happensAt(walking(fomdgt3_7),415).
orientation(fomdgt3_2,42,416). coords(fomdgt3_2,262,96,416).  visible(fomdgt3_2,416). happensAt(inactive(fomdgt3_2),416). orientation(fomdgt3_7,134,416). coords(fomdgt3_7,80,116,416).  visible(fomdgt3_7,416). happensAt(walking(fomdgt3_7),416).
orientation(fomdgt3_2,42,417). coords(fomdgt3_2,262,96,417).  visible(fomdgt3_2,417). happensAt(inactive(fomdgt3_2),417). orientation(fomdgt3_7,134,417). coords(fomdgt3_7,80,116,417).  visible(fomdgt3_7,417). happensAt(walking(fomdgt3_7),417).
orientation(fomdgt3_2,42,418). coords(fomdgt3_2,262,96,418).  visible(fomdgt3_2,418). happensAt(inactive(fomdgt3_2),418). orientation(fomdgt3_7,134,418). coords(fomdgt3_7,80,116,418).  visible(fomdgt3_7,418). happensAt(walking(fomdgt3_7),418).
orientation(fomdgt3_2,42,419). coords(fomdgt3_2,262,96,419).  visible(fomdgt3_2,419). happensAt(inactive(fomdgt3_2),419). orientation(fomdgt3_7,150,419). coords(fomdgt3_7,80,116,419).  visible(fomdgt3_7,419). happensAt(walking(fomdgt3_7),419).
orientation(fomdgt3_2,42,420). coords(fomdgt3_2,262,96,420).  visible(fomdgt3_2,420). happensAt(inactive(fomdgt3_2),420). orientation(fomdgt3_7,150,420). coords(fomdgt3_7,80,116,420).  visible(fomdgt3_7,420). happensAt(walking(fomdgt3_7),420).
orientation(fomdgt3_2,42,421). coords(fomdgt3_2,262,96,421).  visible(fomdgt3_2,421). happensAt(inactive(fomdgt3_2),421). orientation(fomdgt3_7,150,421). coords(fomdgt3_7,80,116,421).  visible(fomdgt3_7,421). happensAt(walking(fomdgt3_7),421).
orientation(fomdgt3_2,42,422). coords(fomdgt3_2,262,96,422).  visible(fomdgt3_2,422). happensAt(inactive(fomdgt3_2),422). orientation(fomdgt3_7,150,422). coords(fomdgt3_7,81,116,422).  visible(fomdgt3_7,422). happensAt(walking(fomdgt3_7),422).
orientation(fomdgt3_2,42,423). coords(fomdgt3_2,262,96,423).  visible(fomdgt3_2,423). happensAt(inactive(fomdgt3_2),423). orientation(fomdgt3_7,150,423). coords(fomdgt3_7,82,117,423).  visible(fomdgt3_7,423). happensAt(walking(fomdgt3_7),423).
orientation(fomdgt3_2,42,424). coords(fomdgt3_2,262,96,424).  visible(fomdgt3_2,424). happensAt(inactive(fomdgt3_2),424). orientation(fomdgt3_7,150,424). coords(fomdgt3_7,82,117,424).  visible(fomdgt3_7,424). happensAt(walking(fomdgt3_7),424).
orientation(fomdgt3_2,42,425). coords(fomdgt3_2,262,96,425).  visible(fomdgt3_2,425). happensAt(inactive(fomdgt3_2),425). orientation(fomdgt3_7,150,425). coords(fomdgt3_7,84,119,425).  visible(fomdgt3_7,425). happensAt(walking(fomdgt3_7),425).
orientation(fomdgt3_2,42,426). coords(fomdgt3_2,262,96,426).  visible(fomdgt3_2,426). happensAt(inactive(fomdgt3_2),426). orientation(fomdgt3_7,150,426). coords(fomdgt3_7,84,121,426).  visible(fomdgt3_7,426). happensAt(walking(fomdgt3_7),426).
orientation(fomdgt3_2,42,427). coords(fomdgt3_2,262,96,427).  visible(fomdgt3_2,427). happensAt(inactive(fomdgt3_2),427). orientation(fomdgt3_7,150,427). coords(fomdgt3_7,84,122,427).  visible(fomdgt3_7,427). happensAt(walking(fomdgt3_7),427).
orientation(fomdgt3_2,42,428). coords(fomdgt3_2,262,96,428).  visible(fomdgt3_2,428). happensAt(inactive(fomdgt3_2),428). orientation(fomdgt3_7,150,428). coords(fomdgt3_7,85,123,428).  visible(fomdgt3_7,428). happensAt(walking(fomdgt3_7),428).
orientation(fomdgt3_2,42,429). coords(fomdgt3_2,262,96,429).  visible(fomdgt3_2,429). happensAt(inactive(fomdgt3_2),429). orientation(fomdgt3_7,150,429). coords(fomdgt3_7,85,124,429).  visible(fomdgt3_7,429). happensAt(walking(fomdgt3_7),429).
orientation(fomdgt3_2,42,430). coords(fomdgt3_2,262,96,430).  visible(fomdgt3_2,430). happensAt(inactive(fomdgt3_2),430). orientation(fomdgt3_7,150,430). coords(fomdgt3_7,86,124,430).  visible(fomdgt3_7,430). happensAt(walking(fomdgt3_7),430).
orientation(fomdgt3_2,42,431). coords(fomdgt3_2,262,96,431).  visible(fomdgt3_2,431). happensAt(inactive(fomdgt3_2),431). orientation(fomdgt3_7,150,431). coords(fomdgt3_7,86,124,431).  visible(fomdgt3_7,431). happensAt(walking(fomdgt3_7),431).
orientation(fomdgt3_2,42,432). coords(fomdgt3_2,262,96,432).  visible(fomdgt3_2,432). happensAt(inactive(fomdgt3_2),432). orientation(fomdgt3_7,150,432). coords(fomdgt3_7,86,125,432).  visible(fomdgt3_7,432). happensAt(walking(fomdgt3_7),432).
orientation(fomdgt3_2,42,433). coords(fomdgt3_2,262,96,433).  visible(fomdgt3_2,433). happensAt(inactive(fomdgt3_2),433). orientation(fomdgt3_7,150,433). coords(fomdgt3_7,86,125,433).  visible(fomdgt3_7,433). happensAt(walking(fomdgt3_7),433).
orientation(fomdgt3_2,42,434). coords(fomdgt3_2,262,96,434).  visible(fomdgt3_2,434). happensAt(inactive(fomdgt3_2),434). orientation(fomdgt3_7,150,434). coords(fomdgt3_7,86,126,434).  visible(fomdgt3_7,434). happensAt(walking(fomdgt3_7),434).
#show holdsAt/2.