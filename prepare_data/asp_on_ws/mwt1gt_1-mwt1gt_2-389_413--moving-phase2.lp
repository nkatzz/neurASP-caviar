

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

	person(mwt1gt_1;mwt1gt_2).
orientation(mwt1gt_1,160,389). coords(mwt1gt_1,96,126,389).  visible(mwt1gt_1,389). orientation(mwt1gt_2,140,389). coords(mwt1gt_2,97,107,389).  visible(mwt1gt_2,389).
orientation(mwt1gt_1,160,390). coords(mwt1gt_1,96,126,390).  visible(mwt1gt_1,390). orientation(mwt1gt_2,140,390). coords(mwt1gt_2,97,107,390).  visible(mwt1gt_2,390).
orientation(mwt1gt_1,160,391). coords(mwt1gt_1,96,126,391).  visible(mwt1gt_1,391). orientation(mwt1gt_2,140,391). coords(mwt1gt_2,97,107,391).  visible(mwt1gt_2,391).
orientation(mwt1gt_1,160,392). coords(mwt1gt_1,96,126,392).  visible(mwt1gt_1,392). orientation(mwt1gt_2,140,392). coords(mwt1gt_2,96,107,392).  visible(mwt1gt_2,392).
orientation(mwt1gt_1,160,393). coords(mwt1gt_1,96,126,393).  visible(mwt1gt_1,393). orientation(mwt1gt_2,140,393). coords(mwt1gt_2,96,107,393).  visible(mwt1gt_2,393).
orientation(mwt1gt_1,160,394). coords(mwt1gt_1,96,126,394).  visible(mwt1gt_1,394). orientation(mwt1gt_2,140,394). coords(mwt1gt_2,96,107,394).  visible(mwt1gt_2,394).
orientation(mwt1gt_1,160,395). coords(mwt1gt_1,96,126,395).  visible(mwt1gt_1,395). orientation(mwt1gt_2,140,395). coords(mwt1gt_2,96,107,395).  visible(mwt1gt_2,395).
orientation(mwt1gt_1,160,396). coords(mwt1gt_1,91,120,396).  visible(mwt1gt_1,396). orientation(mwt1gt_2,140,396). coords(mwt1gt_2,93,105,396).  visible(mwt1gt_2,396).
orientation(mwt1gt_1,160,397). coords(mwt1gt_1,91,120,397).  visible(mwt1gt_1,397). orientation(mwt1gt_2,140,397). coords(mwt1gt_2,93,105,397).  visible(mwt1gt_2,397).
orientation(mwt1gt_1,160,398). coords(mwt1gt_1,91,120,398).  visible(mwt1gt_1,398). orientation(mwt1gt_2,140,398). coords(mwt1gt_2,93,105,398).  visible(mwt1gt_2,398).
orientation(mwt1gt_1,160,399). coords(mwt1gt_1,91,120,399).  visible(mwt1gt_1,399). orientation(mwt1gt_2,140,399). coords(mwt1gt_2,93,105,399).  visible(mwt1gt_2,399).
orientation(mwt1gt_1,160,400). coords(mwt1gt_1,90,118,400).  visible(mwt1gt_1,400). orientation(mwt1gt_2,140,400). coords(mwt1gt_2,90,104,400).  visible(mwt1gt_2,400).
orientation(mwt1gt_1,160,401). coords(mwt1gt_1,90,118,401).  visible(mwt1gt_1,401). orientation(mwt1gt_2,140,401). coords(mwt1gt_2,90,104,401).  visible(mwt1gt_2,401).
orientation(mwt1gt_1,160,402). coords(mwt1gt_1,90,118,402).  visible(mwt1gt_1,402). orientation(mwt1gt_2,140,402). coords(mwt1gt_2,90,104,402).  visible(mwt1gt_2,402).
orientation(mwt1gt_1,160,403). coords(mwt1gt_1,90,118,403).  visible(mwt1gt_1,403). orientation(mwt1gt_2,140,403). coords(mwt1gt_2,90,104,403).  visible(mwt1gt_2,403).
orientation(mwt1gt_1,160,404). coords(mwt1gt_1,89,116,404).  visible(mwt1gt_1,404). orientation(mwt1gt_2,140,404). coords(mwt1gt_2,89,103,404).  visible(mwt1gt_2,404).
orientation(mwt1gt_1,160,405). coords(mwt1gt_1,89,116,405).  visible(mwt1gt_1,405). orientation(mwt1gt_2,140,405). coords(mwt1gt_2,89,103,405).  visible(mwt1gt_2,405).
orientation(mwt1gt_1,160,406). coords(mwt1gt_1,89,116,406).  visible(mwt1gt_1,406). orientation(mwt1gt_2,140,406). coords(mwt1gt_2,89,103,406).  visible(mwt1gt_2,406).
orientation(mwt1gt_1,160,407). coords(mwt1gt_1,89,116,407).  visible(mwt1gt_1,407). orientation(mwt1gt_2,140,407). coords(mwt1gt_2,89,103,407).  visible(mwt1gt_2,407).
orientation(mwt1gt_1,160,408). coords(mwt1gt_1,85,113,408).  visible(mwt1gt_1,408). orientation(mwt1gt_2,140,408). coords(mwt1gt_2,89,103,408).  visible(mwt1gt_2,408).
orientation(mwt1gt_1,160,409). coords(mwt1gt_1,85,113,409).  visible(mwt1gt_1,409). orientation(mwt1gt_2,140,409). coords(mwt1gt_2,89,103,409).  visible(mwt1gt_2,409).
orientation(mwt1gt_1,160,410). coords(mwt1gt_1,85,113,410).  visible(mwt1gt_1,410). orientation(mwt1gt_2,140,410). coords(mwt1gt_2,89,103,410).  visible(mwt1gt_2,410).
orientation(mwt1gt_1,160,411). coords(mwt1gt_1,85,113,411).  visible(mwt1gt_1,411). orientation(mwt1gt_2,140,411). coords(mwt1gt_2,89,103,411).  visible(mwt1gt_2,411).
orientation(mwt1gt_1,160,412). coords(mwt1gt_1,84,112,412).  visible(mwt1gt_1,412). orientation(mwt1gt_2,140,412). coords(mwt1gt_2,89,103,412).  visible(mwt1gt_2,412).
orientation(mwt1gt_1,160,413). coords(mwt1gt_1,84,112,413).  visible(mwt1gt_1,413). orientation(mwt1gt_2,140,413). coords(mwt1gt_2,89,103,413).  visible(mwt1gt_2,413).
orientation(mwt1gt_1,160,414). coords(mwt1gt_1,84,112,414).  visible(mwt1gt_1,414). orientation(mwt1gt_2,140,414). coords(mwt1gt_2,89,103,414).  visible(mwt1gt_2,414).
happensAt(disappear(mwt1gt_1),413). happensAt(disappear(mwt1gt_2),413).
1{simpleEvent(T,mwt1gt_1,active); simpleEvent(T,mwt1gt_1,inactive); simpleEvent(T,mwt1gt_1,walking); simpleEvent(T,mwt1gt_1,running)}1 :- T=389..413.
1{simpleEvent(T,mwt1gt_2,active); simpleEvent(T,mwt1gt_2,inactive); simpleEvent(T,mwt1gt_2,walking); simpleEvent(T,mwt1gt_2,running)}1 :- T=389..413.
:- holdsAt(moving(mwt1gt_1,mwt1gt_2),389). :- holdsAt(moving(mwt1gt_2,mwt1gt_1),389).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),390). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),390).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),391). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),391).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),392). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),392).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),393). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),393).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),394). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),394).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),395). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),395).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),396). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),396).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),397). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),397).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),398). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),398).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),399). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),399).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),400). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),400).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),401). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),401).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),402). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),402).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),403). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),403).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),404). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),404).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),405). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),405).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),406). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),406).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),407). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),407).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),408). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),408).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),409). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),409).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),410). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),410).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),411). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),411).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),412). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),412).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),413). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),413).
:- holdsAt(moving(mwt1gt_1,mwt1gt_2),414). :- holdsAt(moving(mwt1gt_2,mwt1gt_1),414).
:- simpleEvent(T,P,inactive), T>389 , T<413.
:- simpleEvent(T,P,running), T>389 , T<413.

#show simpleEvent/3.