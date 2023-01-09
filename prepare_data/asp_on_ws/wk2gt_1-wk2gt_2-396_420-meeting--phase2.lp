

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

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,396). coords(wk2gt_1,115,53,396).  visible(wk2gt_1,396). orientation(wk2gt_2,141,396). coords(wk2gt_2,98,63,396).  visible(wk2gt_2,396).
orientation(wk2gt_1,125,397). coords(wk2gt_1,115,53,397).  visible(wk2gt_1,397). orientation(wk2gt_2,141,397). coords(wk2gt_2,98,63,397).  visible(wk2gt_2,397).
orientation(wk2gt_1,125,398). coords(wk2gt_1,115,53,398).  visible(wk2gt_1,398). orientation(wk2gt_2,141,398). coords(wk2gt_2,98,63,398).  visible(wk2gt_2,398).
orientation(wk2gt_1,125,399). coords(wk2gt_1,115,53,399).  visible(wk2gt_1,399). orientation(wk2gt_2,141,399). coords(wk2gt_2,98,63,399).  visible(wk2gt_2,399).
orientation(wk2gt_1,125,400). coords(wk2gt_1,115,53,400).  visible(wk2gt_1,400). orientation(wk2gt_2,141,400). coords(wk2gt_2,98,63,400).  visible(wk2gt_2,400).
orientation(wk2gt_1,125,401). coords(wk2gt_1,115,53,401).  visible(wk2gt_1,401). orientation(wk2gt_2,141,401). coords(wk2gt_2,98,63,401).  visible(wk2gt_2,401).
orientation(wk2gt_1,125,402). coords(wk2gt_1,115,53,402).  visible(wk2gt_1,402). orientation(wk2gt_2,141,402). coords(wk2gt_2,98,63,402).  visible(wk2gt_2,402).
orientation(wk2gt_1,125,403). coords(wk2gt_1,115,53,403).  visible(wk2gt_1,403). orientation(wk2gt_2,141,403). coords(wk2gt_2,98,63,403).  visible(wk2gt_2,403).
orientation(wk2gt_1,125,404). coords(wk2gt_1,115,53,404).  visible(wk2gt_1,404). orientation(wk2gt_2,141,404). coords(wk2gt_2,98,63,404).  visible(wk2gt_2,404).
orientation(wk2gt_1,125,405). coords(wk2gt_1,115,53,405).  visible(wk2gt_1,405). orientation(wk2gt_2,141,405). coords(wk2gt_2,98,63,405).  visible(wk2gt_2,405).
orientation(wk2gt_1,125,406). coords(wk2gt_1,115,53,406).  visible(wk2gt_1,406). orientation(wk2gt_2,141,406). coords(wk2gt_2,98,63,406).  visible(wk2gt_2,406).
orientation(wk2gt_1,125,407). coords(wk2gt_1,115,53,407).  visible(wk2gt_1,407). orientation(wk2gt_2,141,407). coords(wk2gt_2,98,63,407).  visible(wk2gt_2,407).
orientation(wk2gt_1,125,408). coords(wk2gt_1,115,53,408).  visible(wk2gt_1,408). orientation(wk2gt_2,141,408). coords(wk2gt_2,98,63,408).  visible(wk2gt_2,408).
orientation(wk2gt_1,125,409). coords(wk2gt_1,115,53,409).  visible(wk2gt_1,409). orientation(wk2gt_2,141,409). coords(wk2gt_2,98,63,409).  visible(wk2gt_2,409).
orientation(wk2gt_1,125,410). coords(wk2gt_1,115,53,410).  visible(wk2gt_1,410). orientation(wk2gt_2,141,410). coords(wk2gt_2,98,63,410).  visible(wk2gt_2,410).
orientation(wk2gt_1,125,411). coords(wk2gt_1,115,53,411).  visible(wk2gt_1,411). orientation(wk2gt_2,141,411). coords(wk2gt_2,98,63,411).  visible(wk2gt_2,411).
orientation(wk2gt_1,125,412). coords(wk2gt_1,115,53,412).  visible(wk2gt_1,412). orientation(wk2gt_2,141,412). coords(wk2gt_2,98,63,412).  visible(wk2gt_2,412).
orientation(wk2gt_1,125,413). coords(wk2gt_1,115,53,413).  visible(wk2gt_1,413). orientation(wk2gt_2,141,413). coords(wk2gt_2,98,63,413).  visible(wk2gt_2,413).
orientation(wk2gt_1,125,414). coords(wk2gt_1,115,53,414).  visible(wk2gt_1,414). orientation(wk2gt_2,141,414). coords(wk2gt_2,98,63,414).  visible(wk2gt_2,414).
orientation(wk2gt_1,125,415). coords(wk2gt_1,115,53,415).  visible(wk2gt_1,415). orientation(wk2gt_2,141,415). coords(wk2gt_2,98,63,415).  visible(wk2gt_2,415).
orientation(wk2gt_1,125,416). coords(wk2gt_1,115,53,416).  visible(wk2gt_1,416). orientation(wk2gt_2,141,416). coords(wk2gt_2,98,63,416).  visible(wk2gt_2,416).
orientation(wk2gt_1,125,417). coords(wk2gt_1,115,53,417).  visible(wk2gt_1,417). orientation(wk2gt_2,141,417). coords(wk2gt_2,98,63,417).  visible(wk2gt_2,417).
orientation(wk2gt_1,125,418). coords(wk2gt_1,115,53,418).  visible(wk2gt_1,418). orientation(wk2gt_2,141,418). coords(wk2gt_2,98,63,418).  visible(wk2gt_2,418).
orientation(wk2gt_1,125,419). coords(wk2gt_1,115,53,419).  visible(wk2gt_1,419). orientation(wk2gt_2,141,419). coords(wk2gt_2,98,63,419).  visible(wk2gt_2,419).
orientation(wk2gt_1,125,420). coords(wk2gt_1,115,53,420).  visible(wk2gt_1,420). orientation(wk2gt_2,141,420). coords(wk2gt_2,98,63,420).  visible(wk2gt_2,420).
orientation(wk2gt_1,125,421). coords(wk2gt_1,115,53,421).  visible(wk2gt_1,421). orientation(wk2gt_2,141,421). coords(wk2gt_2,98,63,421).  visible(wk2gt_2,421).
happensAt(disappear(wk2gt_1),420). happensAt(disappear(wk2gt_2),420).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=396..420.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=396..420.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),396). :- holdsAt(meeting(wk2gt_2,wk2gt_1),396).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),397). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),397).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),398). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),398).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),399). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),399).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),400). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),400).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),401). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),401).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),402). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),402).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),403). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),403).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),404). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),404).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),405). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),405).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),406). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),406).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),407). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),407).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),408). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),408).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),409). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),409).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),410). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),410).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),411). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),411).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),412). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),412).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),413). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),413).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),414). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),414).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),415). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),415).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),416). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),416).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),417). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),417).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),418). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),418).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),419). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),419).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),420). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),420).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),421). :- holdsAt(meeting(wk2gt_2,wk2gt_1),421).
:- simpleEvent(T,P,walking), T>396 , T<420.
:- simpleEvent(T,P,running), T>396 , T<420.

#show simpleEvent/3.