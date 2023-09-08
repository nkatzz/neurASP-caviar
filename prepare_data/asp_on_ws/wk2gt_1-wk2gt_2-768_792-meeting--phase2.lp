

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
orientation(wk2gt_1,125,768). coords(wk2gt_1,115,53,768).  visible(wk2gt_1,768). orientation(wk2gt_2,142,768). coords(wk2gt_2,99,63,768).  visible(wk2gt_2,768).
orientation(wk2gt_1,125,769). coords(wk2gt_1,115,53,769).  visible(wk2gt_1,769). orientation(wk2gt_2,142,769). coords(wk2gt_2,99,63,769).  visible(wk2gt_2,769).
orientation(wk2gt_1,125,770). coords(wk2gt_1,115,53,770).  visible(wk2gt_1,770). orientation(wk2gt_2,142,770). coords(wk2gt_2,99,63,770).  visible(wk2gt_2,770).
orientation(wk2gt_1,125,771). coords(wk2gt_1,115,53,771).  visible(wk2gt_1,771). orientation(wk2gt_2,142,771). coords(wk2gt_2,99,63,771).  visible(wk2gt_2,771).
orientation(wk2gt_1,125,772). coords(wk2gt_1,115,53,772).  visible(wk2gt_1,772). orientation(wk2gt_2,142,772). coords(wk2gt_2,99,63,772).  visible(wk2gt_2,772).
orientation(wk2gt_1,125,773). coords(wk2gt_1,115,53,773).  visible(wk2gt_1,773). orientation(wk2gt_2,142,773). coords(wk2gt_2,99,63,773).  visible(wk2gt_2,773).
orientation(wk2gt_1,125,774). coords(wk2gt_1,115,53,774).  visible(wk2gt_1,774). orientation(wk2gt_2,142,774). coords(wk2gt_2,99,63,774).  visible(wk2gt_2,774).
orientation(wk2gt_1,125,775). coords(wk2gt_1,115,53,775).  visible(wk2gt_1,775). orientation(wk2gt_2,142,775). coords(wk2gt_2,99,63,775).  visible(wk2gt_2,775).
orientation(wk2gt_1,125,776). coords(wk2gt_1,115,53,776).  visible(wk2gt_1,776). orientation(wk2gt_2,142,776). coords(wk2gt_2,99,63,776).  visible(wk2gt_2,776).
orientation(wk2gt_1,125,777). coords(wk2gt_1,115,53,777).  visible(wk2gt_1,777). orientation(wk2gt_2,142,777). coords(wk2gt_2,99,63,777).  visible(wk2gt_2,777).
orientation(wk2gt_1,125,778). coords(wk2gt_1,115,53,778).  visible(wk2gt_1,778). orientation(wk2gt_2,142,778). coords(wk2gt_2,99,63,778).  visible(wk2gt_2,778).
orientation(wk2gt_1,125,779). coords(wk2gt_1,115,53,779).  visible(wk2gt_1,779). orientation(wk2gt_2,142,779). coords(wk2gt_2,99,63,779).  visible(wk2gt_2,779).
orientation(wk2gt_1,125,780). coords(wk2gt_1,115,53,780).  visible(wk2gt_1,780). orientation(wk2gt_2,142,780). coords(wk2gt_2,99,63,780).  visible(wk2gt_2,780).
orientation(wk2gt_1,125,781). coords(wk2gt_1,115,53,781).  visible(wk2gt_1,781). orientation(wk2gt_2,142,781). coords(wk2gt_2,99,63,781).  visible(wk2gt_2,781).
orientation(wk2gt_1,125,782). coords(wk2gt_1,115,53,782).  visible(wk2gt_1,782). orientation(wk2gt_2,142,782). coords(wk2gt_2,99,63,782).  visible(wk2gt_2,782).
orientation(wk2gt_1,125,783). coords(wk2gt_1,115,53,783).  visible(wk2gt_1,783). orientation(wk2gt_2,142,783). coords(wk2gt_2,99,63,783).  visible(wk2gt_2,783).
orientation(wk2gt_1,125,784). coords(wk2gt_1,115,53,784).  visible(wk2gt_1,784). orientation(wk2gt_2,142,784). coords(wk2gt_2,99,63,784).  visible(wk2gt_2,784).
orientation(wk2gt_1,125,785). coords(wk2gt_1,115,53,785).  visible(wk2gt_1,785). orientation(wk2gt_2,142,785). coords(wk2gt_2,99,63,785).  visible(wk2gt_2,785).
orientation(wk2gt_1,125,786). coords(wk2gt_1,115,53,786).  visible(wk2gt_1,786). orientation(wk2gt_2,142,786). coords(wk2gt_2,99,63,786).  visible(wk2gt_2,786).
orientation(wk2gt_1,125,787). coords(wk2gt_1,115,53,787).  visible(wk2gt_1,787). orientation(wk2gt_2,142,787). coords(wk2gt_2,99,63,787).  visible(wk2gt_2,787).
orientation(wk2gt_1,125,788). coords(wk2gt_1,115,53,788).  visible(wk2gt_1,788). orientation(wk2gt_2,142,788). coords(wk2gt_2,99,63,788).  visible(wk2gt_2,788).
orientation(wk2gt_1,125,789). coords(wk2gt_1,115,53,789).  visible(wk2gt_1,789). orientation(wk2gt_2,142,789). coords(wk2gt_2,99,63,789).  visible(wk2gt_2,789).
orientation(wk2gt_1,125,790). coords(wk2gt_1,115,53,790).  visible(wk2gt_1,790). orientation(wk2gt_2,142,790). coords(wk2gt_2,99,63,790).  visible(wk2gt_2,790).
orientation(wk2gt_1,125,791). coords(wk2gt_1,115,53,791).  visible(wk2gt_1,791). orientation(wk2gt_2,142,791). coords(wk2gt_2,99,63,791).  visible(wk2gt_2,791).
orientation(wk2gt_1,125,792). coords(wk2gt_1,115,53,792).  visible(wk2gt_1,792). orientation(wk2gt_2,142,792). coords(wk2gt_2,99,63,792).  visible(wk2gt_2,792).
orientation(wk2gt_1,125,793). coords(wk2gt_1,115,53,793).  visible(wk2gt_1,793). orientation(wk2gt_2,142,793). coords(wk2gt_2,99,63,793).  visible(wk2gt_2,793).
happensAt(disappear(wk2gt_1),792). happensAt(disappear(wk2gt_2),792).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=768..792.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=768..792.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),768). :- holdsAt(meeting(wk2gt_2,wk2gt_1),768).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),769). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),769).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),770). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),770).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),771). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),771).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),772). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),772).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),773). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),773).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),774). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),774).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),775). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),775).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),776). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),776).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),777). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),777).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),778). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),778).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),779). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),779).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),780). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),780).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),781). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),781).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),782). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),782).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),783). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),783).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),784). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),784).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),785). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),785).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),786). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),786).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),787). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),787).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),788). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),788).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),789). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),789).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),790). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),790).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),791). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),791).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),792). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),792).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),793). :- holdsAt(meeting(wk2gt_2,wk2gt_1),793).
:- simpleEvent(T,P,walking), T>768 , T<792.
:- simpleEvent(T,P,running), T>768 , T<792.

#show simpleEvent/3.