

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,153,385). coords(mwt2gt_1,118,85,385).  visible(mwt2gt_1,385). orientation(mwt2gt_2,151,385). coords(mwt2gt_2,115,104,385).  visible(mwt2gt_2,385).
orientation(mwt2gt_1,153,386). coords(mwt2gt_1,117,85,386).  visible(mwt2gt_1,386). orientation(mwt2gt_2,151,386). coords(mwt2gt_2,115,105,386).  visible(mwt2gt_2,386).
orientation(mwt2gt_1,153,387). coords(mwt2gt_1,116,85,387).  visible(mwt2gt_1,387). orientation(mwt2gt_2,151,387). coords(mwt2gt_2,115,107,387).  visible(mwt2gt_2,387).
orientation(mwt2gt_1,153,388). coords(mwt2gt_1,115,85,388).  visible(mwt2gt_1,388). orientation(mwt2gt_2,144,388). coords(mwt2gt_2,114,108,388).  visible(mwt2gt_2,388).
orientation(mwt2gt_1,153,389). coords(mwt2gt_1,113,86,389).  visible(mwt2gt_1,389). orientation(mwt2gt_2,144,389). coords(mwt2gt_2,114,109,389).  visible(mwt2gt_2,389).
orientation(mwt2gt_1,149,390). coords(mwt2gt_1,112,86,390).  visible(mwt2gt_1,390). orientation(mwt2gt_2,144,390). coords(mwt2gt_2,113,110,390).  visible(mwt2gt_2,390).
orientation(mwt2gt_1,149,391). coords(mwt2gt_1,111,87,391).  visible(mwt2gt_1,391). orientation(mwt2gt_2,144,391). coords(mwt2gt_2,113,110,391).  visible(mwt2gt_2,391).
orientation(mwt2gt_1,149,392). coords(mwt2gt_1,111,88,392).  visible(mwt2gt_1,392). orientation(mwt2gt_2,144,392). coords(mwt2gt_2,112,110,392).  visible(mwt2gt_2,392).
orientation(mwt2gt_1,149,393). coords(mwt2gt_1,110,88,393).  visible(mwt2gt_1,393). orientation(mwt2gt_2,144,393). coords(mwt2gt_2,111,111,393).  visible(mwt2gt_2,393).
orientation(mwt2gt_1,146,394). coords(mwt2gt_1,111,88,394).  visible(mwt2gt_1,394). orientation(mwt2gt_2,144,394). coords(mwt2gt_2,110,112,394).  visible(mwt2gt_2,394).
orientation(mwt2gt_1,146,395). coords(mwt2gt_1,110,89,395).  visible(mwt2gt_1,395). orientation(mwt2gt_2,144,395). coords(mwt2gt_2,108,112,395).  visible(mwt2gt_2,395).
orientation(mwt2gt_1,146,396). coords(mwt2gt_1,110,89,396).  visible(mwt2gt_1,396). orientation(mwt2gt_2,144,396). coords(mwt2gt_2,106,112,396).  visible(mwt2gt_2,396).
orientation(mwt2gt_1,146,397). coords(mwt2gt_1,109,89,397).  visible(mwt2gt_1,397). orientation(mwt2gt_2,144,397). coords(mwt2gt_2,104,113,397).  visible(mwt2gt_2,397).
orientation(mwt2gt_1,146,398). coords(mwt2gt_1,108,90,398).  visible(mwt2gt_1,398). orientation(mwt2gt_2,144,398). coords(mwt2gt_2,104,113,398).  visible(mwt2gt_2,398).
orientation(mwt2gt_1,146,399). coords(mwt2gt_1,108,90,399).  visible(mwt2gt_1,399). orientation(mwt2gt_2,144,399). coords(mwt2gt_2,103,113,399).  visible(mwt2gt_2,399).
orientation(mwt2gt_1,146,400). coords(mwt2gt_1,107,90,400).  visible(mwt2gt_1,400). orientation(mwt2gt_2,144,400). coords(mwt2gt_2,102,113,400).  visible(mwt2gt_2,400).
orientation(mwt2gt_1,146,401). coords(mwt2gt_1,105,90,401).  visible(mwt2gt_1,401). orientation(mwt2gt_2,144,401). coords(mwt2gt_2,102,113,401).  visible(mwt2gt_2,401).
orientation(mwt2gt_1,150,402). coords(mwt2gt_1,105,90,402).  visible(mwt2gt_1,402). orientation(mwt2gt_2,144,402). coords(mwt2gt_2,102,113,402).  visible(mwt2gt_2,402).
orientation(mwt2gt_1,150,403). coords(mwt2gt_1,103,90,403).  visible(mwt2gt_1,403). orientation(mwt2gt_2,149,403). coords(mwt2gt_2,102,114,403).  visible(mwt2gt_2,403).
orientation(mwt2gt_1,150,404). coords(mwt2gt_1,102,90,404).  visible(mwt2gt_1,404). orientation(mwt2gt_2,149,404). coords(mwt2gt_2,101,114,404).  visible(mwt2gt_2,404).
orientation(mwt2gt_1,150,405). coords(mwt2gt_1,102,90,405).  visible(mwt2gt_1,405). orientation(mwt2gt_2,149,405). coords(mwt2gt_2,100,114,405).  visible(mwt2gt_2,405).
orientation(mwt2gt_1,150,406). coords(mwt2gt_1,101,90,406).  visible(mwt2gt_1,406). orientation(mwt2gt_2,149,406). coords(mwt2gt_2,100,114,406).  visible(mwt2gt_2,406).
orientation(mwt2gt_1,150,407). coords(mwt2gt_1,101,90,407).  visible(mwt2gt_1,407). orientation(mwt2gt_2,149,407). coords(mwt2gt_2,98,114,407).  visible(mwt2gt_2,407).
orientation(mwt2gt_1,150,408). coords(mwt2gt_1,101,90,408).  visible(mwt2gt_1,408). orientation(mwt2gt_2,149,408). coords(mwt2gt_2,97,115,408).  visible(mwt2gt_2,408).
orientation(mwt2gt_1,150,409). coords(mwt2gt_1,101,90,409).  visible(mwt2gt_1,409). orientation(mwt2gt_2,149,409). coords(mwt2gt_2,96,115,409).  visible(mwt2gt_2,409).
orientation(mwt2gt_1,150,410). coords(mwt2gt_1,101,90,410).  visible(mwt2gt_1,410). orientation(mwt2gt_2,149,410). coords(mwt2gt_2,96,115,410).  visible(mwt2gt_2,410).
happensAt(disappear(mwt2gt_1),409). happensAt(disappear(mwt2gt_2),409).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=385..409.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=385..409.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),385). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),385).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),386). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),386).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),387). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),387).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),388). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),388).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),389). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),389).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),390). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),390).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),391). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),391).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),392). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),392).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),393). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),393).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),394). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),394).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),395). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),395).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),396). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),396).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),397). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),397).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),398). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),398).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),399). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),399).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),400). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),400).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),401). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),401).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),402). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),402).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),403). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),403).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),404). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),404).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),405). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),405).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),406). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),406).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),407). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),407).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),408). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),408).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),409). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),409).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),410). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),410).
:- simpleEvent(T,P,inactive), T>385 , T<409.
:- simpleEvent(T,P,running), T>385 , T<409.

#show simpleEvent/3.