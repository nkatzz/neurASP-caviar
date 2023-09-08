

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
orientation(wk2gt_1,125,384). coords(wk2gt_1,115,53,384).  visible(wk2gt_1,384). orientation(wk2gt_2,141,384). coords(wk2gt_2,98,63,384).  visible(wk2gt_2,384).
orientation(wk2gt_1,125,385). coords(wk2gt_1,115,53,385).  visible(wk2gt_1,385). orientation(wk2gt_2,141,385). coords(wk2gt_2,98,63,385).  visible(wk2gt_2,385).
orientation(wk2gt_1,125,386). coords(wk2gt_1,115,53,386).  visible(wk2gt_1,386). orientation(wk2gt_2,141,386). coords(wk2gt_2,98,63,386).  visible(wk2gt_2,386).
orientation(wk2gt_1,125,387). coords(wk2gt_1,115,53,387).  visible(wk2gt_1,387). orientation(wk2gt_2,141,387). coords(wk2gt_2,98,63,387).  visible(wk2gt_2,387).
orientation(wk2gt_1,125,388). coords(wk2gt_1,115,53,388).  visible(wk2gt_1,388). orientation(wk2gt_2,141,388). coords(wk2gt_2,98,63,388).  visible(wk2gt_2,388).
orientation(wk2gt_1,125,389). coords(wk2gt_1,115,53,389).  visible(wk2gt_1,389). orientation(wk2gt_2,141,389). coords(wk2gt_2,98,63,389).  visible(wk2gt_2,389).
orientation(wk2gt_1,125,390). coords(wk2gt_1,115,53,390).  visible(wk2gt_1,390). orientation(wk2gt_2,141,390). coords(wk2gt_2,98,63,390).  visible(wk2gt_2,390).
orientation(wk2gt_1,125,391). coords(wk2gt_1,115,53,391).  visible(wk2gt_1,391). orientation(wk2gt_2,141,391). coords(wk2gt_2,98,63,391).  visible(wk2gt_2,391).
orientation(wk2gt_1,125,392). coords(wk2gt_1,115,53,392).  visible(wk2gt_1,392). orientation(wk2gt_2,141,392). coords(wk2gt_2,98,63,392).  visible(wk2gt_2,392).
orientation(wk2gt_1,125,393). coords(wk2gt_1,115,53,393).  visible(wk2gt_1,393). orientation(wk2gt_2,141,393). coords(wk2gt_2,98,63,393).  visible(wk2gt_2,393).
orientation(wk2gt_1,125,394). coords(wk2gt_1,115,53,394).  visible(wk2gt_1,394). orientation(wk2gt_2,141,394). coords(wk2gt_2,98,63,394).  visible(wk2gt_2,394).
orientation(wk2gt_1,125,395). coords(wk2gt_1,115,53,395).  visible(wk2gt_1,395). orientation(wk2gt_2,141,395). coords(wk2gt_2,98,63,395).  visible(wk2gt_2,395).
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
happensAt(disappear(wk2gt_1),408). happensAt(disappear(wk2gt_2),408).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=384..408.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=384..408.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),384). :- holdsAt(meeting(wk2gt_2,wk2gt_1),384).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),385). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),385).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),386). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),386).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),387). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),387).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),388). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),388).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),389). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),389).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),390). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),390).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),391). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),391).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),392). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),392).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),393). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),393).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),394). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),394).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),395). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),395).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),396). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),396).
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
:- holdsAt(meeting(wk2gt_1,wk2gt_2),409). :- holdsAt(meeting(wk2gt_2,wk2gt_1),409).
:- simpleEvent(T,P,walking), T>384 , T<408.
:- simpleEvent(T,P,running), T>384 , T<408.

#show simpleEvent/3.