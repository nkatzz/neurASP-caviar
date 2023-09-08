

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
orientation(mwt2gt_1,153,553). coords(mwt2gt_1,38,155,553).  visible(mwt2gt_1,553). orientation(mwt2gt_2,152,553). coords(mwt2gt_2,33,167,553).  visible(mwt2gt_2,553).
orientation(mwt2gt_1,153,554). coords(mwt2gt_1,38,155,554).  visible(mwt2gt_1,554). orientation(mwt2gt_2,152,554). coords(mwt2gt_2,33,167,554).  visible(mwt2gt_2,554).
orientation(mwt2gt_1,153,555). coords(mwt2gt_1,38,155,555).  visible(mwt2gt_1,555). orientation(mwt2gt_2,152,555). coords(mwt2gt_2,32,167,555).  visible(mwt2gt_2,555).
orientation(mwt2gt_1,153,556). coords(mwt2gt_1,38,155,556).  visible(mwt2gt_1,556). orientation(mwt2gt_2,152,556). coords(mwt2gt_2,32,167,556).  visible(mwt2gt_2,556).
orientation(mwt2gt_1,153,557). coords(mwt2gt_1,37,155,557).  visible(mwt2gt_1,557). orientation(mwt2gt_2,152,557). coords(mwt2gt_2,32,167,557).  visible(mwt2gt_2,557).
orientation(mwt2gt_1,153,558). coords(mwt2gt_1,36,155,558).  visible(mwt2gt_1,558). orientation(mwt2gt_2,152,558). coords(mwt2gt_2,32,167,558).  visible(mwt2gt_2,558).
orientation(mwt2gt_1,153,559). coords(mwt2gt_1,36,155,559).  visible(mwt2gt_1,559). orientation(mwt2gt_2,152,559). coords(mwt2gt_2,32,167,559).  visible(mwt2gt_2,559).
orientation(mwt2gt_1,153,560). coords(mwt2gt_1,36,155,560).  visible(mwt2gt_1,560). orientation(mwt2gt_2,152,560). coords(mwt2gt_2,32,167,560).  visible(mwt2gt_2,560).
orientation(mwt2gt_1,153,561). coords(mwt2gt_1,36,155,561).  visible(mwt2gt_1,561). orientation(mwt2gt_2,152,561). coords(mwt2gt_2,32,167,561).  visible(mwt2gt_2,561).
orientation(mwt2gt_1,153,562). coords(mwt2gt_1,36,155,562).  visible(mwt2gt_1,562). orientation(mwt2gt_2,152,562). coords(mwt2gt_2,32,167,562).  visible(mwt2gt_2,562).
orientation(mwt2gt_1,153,563). coords(mwt2gt_1,36,155,563).  visible(mwt2gt_1,563). orientation(mwt2gt_2,152,563). coords(mwt2gt_2,32,167,563).  visible(mwt2gt_2,563).
orientation(mwt2gt_1,153,564). coords(mwt2gt_1,36,155,564).  visible(mwt2gt_1,564). orientation(mwt2gt_2,152,564). coords(mwt2gt_2,31,167,564).  visible(mwt2gt_2,564).
orientation(mwt2gt_1,153,565). coords(mwt2gt_1,36,155,565).  visible(mwt2gt_1,565). orientation(mwt2gt_2,152,565). coords(mwt2gt_2,31,167,565).  visible(mwt2gt_2,565).
orientation(mwt2gt_1,153,566). coords(mwt2gt_1,35,155,566).  visible(mwt2gt_1,566). orientation(mwt2gt_2,152,566). coords(mwt2gt_2,31,167,566).  visible(mwt2gt_2,566).
orientation(mwt2gt_1,153,567). coords(mwt2gt_1,35,155,567).  visible(mwt2gt_1,567). orientation(mwt2gt_2,152,567). coords(mwt2gt_2,31,167,567).  visible(mwt2gt_2,567).
orientation(mwt2gt_1,153,568). coords(mwt2gt_1,34,155,568).  visible(mwt2gt_1,568). orientation(mwt2gt_2,152,568). coords(mwt2gt_2,29,167,568).  visible(mwt2gt_2,568).
orientation(mwt2gt_1,153,569). coords(mwt2gt_1,34,155,569).  visible(mwt2gt_1,569). orientation(mwt2gt_2,152,569). coords(mwt2gt_2,29,167,569).  visible(mwt2gt_2,569).
orientation(mwt2gt_1,153,570). coords(mwt2gt_1,33,155,570).  visible(mwt2gt_1,570). orientation(mwt2gt_2,152,570). coords(mwt2gt_2,29,167,570).  visible(mwt2gt_2,570).
orientation(mwt2gt_1,153,571). coords(mwt2gt_1,33,155,571).  visible(mwt2gt_1,571). orientation(mwt2gt_2,152,571). coords(mwt2gt_2,29,167,571).  visible(mwt2gt_2,571).
orientation(mwt2gt_1,153,572). coords(mwt2gt_1,33,155,572).  visible(mwt2gt_1,572). orientation(mwt2gt_2,152,572). coords(mwt2gt_2,29,167,572).  visible(mwt2gt_2,572).
orientation(mwt2gt_1,153,573). coords(mwt2gt_1,33,155,573).  visible(mwt2gt_1,573). orientation(mwt2gt_2,152,573). coords(mwt2gt_2,29,167,573).  visible(mwt2gt_2,573).
orientation(mwt2gt_1,153,574). coords(mwt2gt_1,33,155,574).  visible(mwt2gt_1,574). orientation(mwt2gt_2,152,574). coords(mwt2gt_2,28,167,574).  visible(mwt2gt_2,574).
orientation(mwt2gt_1,153,575). coords(mwt2gt_1,33,155,575).  visible(mwt2gt_1,575). orientation(mwt2gt_2,152,575). coords(mwt2gt_2,28,166,575).  visible(mwt2gt_2,575).
orientation(mwt2gt_1,153,576). coords(mwt2gt_1,33,155,576).  visible(mwt2gt_1,576). orientation(mwt2gt_2,152,576). coords(mwt2gt_2,28,166,576).  visible(mwt2gt_2,576).
orientation(mwt2gt_1,153,577). coords(mwt2gt_1,32,155,577).  visible(mwt2gt_1,577). orientation(mwt2gt_2,152,577). coords(mwt2gt_2,28,166,577).  visible(mwt2gt_2,577).
orientation(mwt2gt_1,153,578). coords(mwt2gt_1,32,155,578).  visible(mwt2gt_1,578). orientation(mwt2gt_2,152,578). coords(mwt2gt_2,28,166,578).  visible(mwt2gt_2,578).
happensAt(disappear(mwt2gt_1),577). happensAt(disappear(mwt2gt_2),577).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=553..577.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=553..577.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),553). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),553).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),554). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),554).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),555). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),555).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),556). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),556).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),557). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),557).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),558). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),558).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),559). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),559).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),560). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),560).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),561). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),561).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),562). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),562).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),563). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),563).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),564). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),564).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),565). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),565).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),566). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),566).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),567). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),567).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),568). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),568).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),569). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),569).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),570). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),570).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),571). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),571).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),572). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),572).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),573). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),573).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),574). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),574).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),575). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),575).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),576). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),576).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),577). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),577).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),578). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),578).
:- simpleEvent(T,P,inactive), T>553 , T<577.
:- simpleEvent(T,P,running), T>553 , T<577.

#show simpleEvent/3.