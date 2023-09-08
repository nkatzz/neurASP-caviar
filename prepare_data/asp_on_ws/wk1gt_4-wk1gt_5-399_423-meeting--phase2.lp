

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,163,399). coords(wk1gt_4,52,176,399).  visible(wk1gt_4,399). orientation(wk1gt_5,162,399). coords(wk1gt_5,50,163,399).  visible(wk1gt_5,399).
orientation(wk1gt_4,162,400). coords(wk1gt_4,52,176,400).  visible(wk1gt_4,400). orientation(wk1gt_5,162,400). coords(wk1gt_5,50,163,400).  visible(wk1gt_5,400).
orientation(wk1gt_4,168,401). coords(wk1gt_4,52,176,401).  visible(wk1gt_4,401). orientation(wk1gt_5,162,401). coords(wk1gt_5,50,163,401).  visible(wk1gt_5,401).
orientation(wk1gt_4,162,402). coords(wk1gt_4,52,176,402).  visible(wk1gt_4,402). orientation(wk1gt_5,162,402). coords(wk1gt_5,50,163,402).  visible(wk1gt_5,402).
orientation(wk1gt_4,162,403). coords(wk1gt_4,52,176,403).  visible(wk1gt_4,403). orientation(wk1gt_5,162,403). coords(wk1gt_5,50,163,403).  visible(wk1gt_5,403).
orientation(wk1gt_4,160,404). coords(wk1gt_4,52,176,404).  visible(wk1gt_4,404). orientation(wk1gt_5,162,404). coords(wk1gt_5,50,163,404).  visible(wk1gt_5,404).
orientation(wk1gt_4,154,405). coords(wk1gt_4,52,176,405).  visible(wk1gt_4,405). orientation(wk1gt_5,162,405). coords(wk1gt_5,50,163,405).  visible(wk1gt_5,405).
orientation(wk1gt_4,156,406). coords(wk1gt_4,52,176,406).  visible(wk1gt_4,406). orientation(wk1gt_5,162,406). coords(wk1gt_5,50,163,406).  visible(wk1gt_5,406).
orientation(wk1gt_4,159,407). coords(wk1gt_4,52,176,407).  visible(wk1gt_4,407). orientation(wk1gt_5,162,407). coords(wk1gt_5,50,163,407).  visible(wk1gt_5,407).
orientation(wk1gt_4,156,408). coords(wk1gt_4,52,176,408).  visible(wk1gt_4,408). orientation(wk1gt_5,162,408). coords(wk1gt_5,50,163,408).  visible(wk1gt_5,408).
orientation(wk1gt_4,155,409). coords(wk1gt_4,52,176,409).  visible(wk1gt_4,409). orientation(wk1gt_5,162,409). coords(wk1gt_5,50,163,409).  visible(wk1gt_5,409).
orientation(wk1gt_4,162,410). coords(wk1gt_4,52,176,410).  visible(wk1gt_4,410). orientation(wk1gt_5,162,410). coords(wk1gt_5,50,163,410).  visible(wk1gt_5,410).
orientation(wk1gt_4,168,411). coords(wk1gt_4,52,176,411).  visible(wk1gt_4,411). orientation(wk1gt_5,162,411). coords(wk1gt_5,50,163,411).  visible(wk1gt_5,411).
orientation(wk1gt_4,162,412). coords(wk1gt_4,52,176,412).  visible(wk1gt_4,412). orientation(wk1gt_5,162,412). coords(wk1gt_5,50,163,412).  visible(wk1gt_5,412).
orientation(wk1gt_4,162,413). coords(wk1gt_4,52,176,413).  visible(wk1gt_4,413). orientation(wk1gt_5,162,413). coords(wk1gt_5,50,163,413).  visible(wk1gt_5,413).
orientation(wk1gt_4,161,414). coords(wk1gt_4,52,176,414).  visible(wk1gt_4,414). orientation(wk1gt_5,162,414). coords(wk1gt_5,50,163,414).  visible(wk1gt_5,414).
orientation(wk1gt_4,163,415). coords(wk1gt_4,52,176,415).  visible(wk1gt_4,415). orientation(wk1gt_5,162,415). coords(wk1gt_5,50,163,415).  visible(wk1gt_5,415).
orientation(wk1gt_4,163,416). coords(wk1gt_4,53,176,416).  visible(wk1gt_4,416). orientation(wk1gt_5,162,416). coords(wk1gt_5,50,163,416).  visible(wk1gt_5,416).
orientation(wk1gt_4,160,417). coords(wk1gt_4,53,175,417).  visible(wk1gt_4,417). orientation(wk1gt_5,162,417). coords(wk1gt_5,50,163,417).  visible(wk1gt_5,417).
orientation(wk1gt_4,159,418). coords(wk1gt_4,53,175,418).  visible(wk1gt_4,418). orientation(wk1gt_5,162,418). coords(wk1gt_5,50,163,418).  visible(wk1gt_5,418).
orientation(wk1gt_4,157,419). coords(wk1gt_4,54,175,419).  visible(wk1gt_4,419). orientation(wk1gt_5,162,419). coords(wk1gt_5,50,163,419).  visible(wk1gt_5,419).
orientation(wk1gt_4,157,420). coords(wk1gt_4,54,175,420).  visible(wk1gt_4,420). orientation(wk1gt_5,162,420). coords(wk1gt_5,50,163,420).  visible(wk1gt_5,420).
orientation(wk1gt_4,157,421). coords(wk1gt_4,54,175,421).  visible(wk1gt_4,421). orientation(wk1gt_5,162,421). coords(wk1gt_5,50,163,421).  visible(wk1gt_5,421).
orientation(wk1gt_4,157,422). coords(wk1gt_4,54,174,422).  visible(wk1gt_4,422). orientation(wk1gt_5,162,422). coords(wk1gt_5,50,163,422).  visible(wk1gt_5,422).
orientation(wk1gt_4,157,423). coords(wk1gt_4,54,174,423).  visible(wk1gt_4,423). orientation(wk1gt_5,162,423). coords(wk1gt_5,50,163,423).  visible(wk1gt_5,423).
orientation(wk1gt_4,157,424). coords(wk1gt_4,54,174,424).  visible(wk1gt_4,424). orientation(wk1gt_5,162,424). coords(wk1gt_5,50,163,424).  visible(wk1gt_5,424).
happensAt(disappear(wk1gt_4),423). happensAt(disappear(wk1gt_5),423).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=399..423.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=399..423.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),399). :- holdsAt(meeting(wk1gt_5,wk1gt_4),399).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),400). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),400).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),401). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),401).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),402). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),402).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),403). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),403).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),404). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),404).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),405). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),405).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),406). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),406).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),407). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),407).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),408). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),408).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),409). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),409).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),410). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),410).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),411). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),411).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),412). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),412).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),413). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),413).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),414). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),414).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),415). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),415).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),416). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),416).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),417). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),417).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),418). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),418).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),419). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),419).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),420). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),420).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),421). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),421).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),422). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),422).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),423). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),423).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),424). :- holdsAt(meeting(wk1gt_5,wk1gt_4),424).
:- simpleEvent(T,P,walking), T>399 , T<423.
:- simpleEvent(T,P,running), T>399 , T<423.

#show simpleEvent/3.