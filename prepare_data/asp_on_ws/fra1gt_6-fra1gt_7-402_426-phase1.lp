
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

	orientation(fra1gt_6,160,402). coords(fra1gt_6,117,113,402).  visible(fra1gt_6,402). happensAt(running(fra1gt_6),402). orientation(fra1gt_7,146,402). coords(fra1gt_7,102,78,402).  visible(fra1gt_7,402). happensAt(running(fra1gt_7),402).
orientation(fra1gt_6,160,403). coords(fra1gt_6,115,115,403).  visible(fra1gt_6,403). happensAt(running(fra1gt_6),403). orientation(fra1gt_7,146,403). coords(fra1gt_7,100,78,403).  visible(fra1gt_7,403). happensAt(running(fra1gt_7),403).
orientation(fra1gt_6,160,404). coords(fra1gt_6,111,115,404).  visible(fra1gt_6,404). happensAt(running(fra1gt_6),404). orientation(fra1gt_7,146,404). coords(fra1gt_7,99,78,404).  visible(fra1gt_7,404). happensAt(running(fra1gt_7),404).
orientation(fra1gt_6,160,405). coords(fra1gt_6,108,115,405).  visible(fra1gt_6,405). happensAt(running(fra1gt_6),405). orientation(fra1gt_7,146,405). coords(fra1gt_7,98,78,405).  visible(fra1gt_7,405). happensAt(running(fra1gt_7),405).
orientation(fra1gt_6,164,406). coords(fra1gt_6,106,116,406).  visible(fra1gt_6,406). happensAt(running(fra1gt_6),406). orientation(fra1gt_7,146,406). coords(fra1gt_7,96,78,406).  visible(fra1gt_7,406). happensAt(running(fra1gt_7),406).
orientation(fra1gt_6,167,407). coords(fra1gt_6,105,116,407).  visible(fra1gt_6,407). happensAt(running(fra1gt_6),407). orientation(fra1gt_7,146,407). coords(fra1gt_7,94,78,407).  visible(fra1gt_7,407). happensAt(running(fra1gt_7),407).
orientation(fra1gt_6,167,408). coords(fra1gt_6,103,116,408).  visible(fra1gt_6,408). happensAt(running(fra1gt_6),408). orientation(fra1gt_7,146,408). coords(fra1gt_7,93,79,408).  visible(fra1gt_7,408). happensAt(running(fra1gt_7),408).
orientation(fra1gt_6,167,409). coords(fra1gt_6,101,117,409).  visible(fra1gt_6,409). happensAt(running(fra1gt_6),409). orientation(fra1gt_7,146,409). coords(fra1gt_7,91,79,409).  visible(fra1gt_7,409). happensAt(running(fra1gt_7),409).
orientation(fra1gt_6,167,410). coords(fra1gt_6,100,121,410).  visible(fra1gt_6,410). happensAt(running(fra1gt_6),410). orientation(fra1gt_7,146,410). coords(fra1gt_7,89,79,410).  visible(fra1gt_7,410). happensAt(running(fra1gt_7),410).
orientation(fra1gt_6,167,411). coords(fra1gt_6,97,123,411).  visible(fra1gt_6,411). happensAt(running(fra1gt_6),411). orientation(fra1gt_7,146,411). coords(fra1gt_7,88,79,411).  visible(fra1gt_7,411). happensAt(running(fra1gt_7),411).
orientation(fra1gt_6,173,412). coords(fra1gt_6,95,125,412).  visible(fra1gt_6,412). happensAt(running(fra1gt_6),412). orientation(fra1gt_7,146,412). coords(fra1gt_7,88,79,412).  visible(fra1gt_7,412). happensAt(running(fra1gt_7),412).
orientation(fra1gt_6,173,413). coords(fra1gt_6,92,126,413).  visible(fra1gt_6,413). happensAt(running(fra1gt_6),413). orientation(fra1gt_7,146,413). coords(fra1gt_7,87,78,413).  visible(fra1gt_7,413). happensAt(running(fra1gt_7),413).
orientation(fra1gt_6,173,414). coords(fra1gt_6,90,128,414).  visible(fra1gt_6,414). happensAt(running(fra1gt_6),414). orientation(fra1gt_7,146,414). coords(fra1gt_7,85,79,414).  visible(fra1gt_7,414). happensAt(running(fra1gt_7),414).
orientation(fra1gt_6,173,415). coords(fra1gt_6,88,129,415).  visible(fra1gt_6,415). happensAt(running(fra1gt_6),415). orientation(fra1gt_7,146,415). coords(fra1gt_7,84,79,415).  visible(fra1gt_7,415). happensAt(running(fra1gt_7),415).
orientation(fra1gt_6,173,416). coords(fra1gt_6,86,131,416).  visible(fra1gt_6,416). happensAt(running(fra1gt_6),416). orientation(fra1gt_7,146,416). coords(fra1gt_7,82,79,416).  visible(fra1gt_7,416). happensAt(running(fra1gt_7),416).
orientation(fra1gt_6,173,417). coords(fra1gt_6,85,132,417).  visible(fra1gt_6,417). happensAt(running(fra1gt_6),417). orientation(fra1gt_7,146,417). coords(fra1gt_7,82,79,417).  visible(fra1gt_7,417). happensAt(running(fra1gt_7),417).
orientation(fra1gt_6,173,418). coords(fra1gt_6,84,134,418).  visible(fra1gt_6,418). happensAt(running(fra1gt_6),418). orientation(fra1gt_7,146,418). coords(fra1gt_7,80,79,418).  visible(fra1gt_7,418). happensAt(running(fra1gt_7),418).
orientation(fra1gt_6,173,419). coords(fra1gt_6,84,136,419).  visible(fra1gt_6,419). happensAt(running(fra1gt_6),419). orientation(fra1gt_7,146,419). coords(fra1gt_7,79,79,419).  visible(fra1gt_7,419). happensAt(running(fra1gt_7),419).
orientation(fra1gt_6,173,420). coords(fra1gt_6,82,138,420).  visible(fra1gt_6,420). happensAt(running(fra1gt_6),420). orientation(fra1gt_7,146,420). coords(fra1gt_7,79,80,420).  visible(fra1gt_7,420). happensAt(running(fra1gt_7),420).
orientation(fra1gt_6,173,421). coords(fra1gt_6,79,139,421).  visible(fra1gt_6,421). happensAt(running(fra1gt_6),421). orientation(fra1gt_7,146,421). coords(fra1gt_7,78,79,421).  visible(fra1gt_7,421). happensAt(running(fra1gt_7),421).
orientation(fra1gt_6,173,422). coords(fra1gt_6,78,141,422).  visible(fra1gt_6,422). happensAt(running(fra1gt_6),422). orientation(fra1gt_7,146,422). coords(fra1gt_7,78,80,422).  visible(fra1gt_7,422). happensAt(running(fra1gt_7),422).
orientation(fra1gt_6,173,423). coords(fra1gt_6,76,142,423).  visible(fra1gt_6,423). happensAt(running(fra1gt_6),423). orientation(fra1gt_7,146,423). coords(fra1gt_7,76,80,423).  visible(fra1gt_7,423). happensAt(running(fra1gt_7),423).
orientation(fra1gt_6,173,424). coords(fra1gt_6,73,142,424).  visible(fra1gt_6,424). happensAt(running(fra1gt_6),424). orientation(fra1gt_7,146,424). coords(fra1gt_7,76,80,424).  visible(fra1gt_7,424). happensAt(running(fra1gt_7),424).
orientation(fra1gt_6,173,425). coords(fra1gt_6,72,143,425).  visible(fra1gt_6,425). happensAt(running(fra1gt_6),425). orientation(fra1gt_7,146,425). coords(fra1gt_7,75,80,425).  visible(fra1gt_7,425). happensAt(running(fra1gt_7),425).
orientation(fra1gt_6,177,426). coords(fra1gt_6,71,144,426).  visible(fra1gt_6,426). happensAt(running(fra1gt_6),426). orientation(fra1gt_7,146,426). coords(fra1gt_7,74,81,426).  visible(fra1gt_7,426). happensAt(running(fra1gt_7),426).
#show holdsAt/2.