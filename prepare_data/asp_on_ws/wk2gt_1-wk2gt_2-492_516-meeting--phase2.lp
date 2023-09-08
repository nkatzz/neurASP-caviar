

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
orientation(wk2gt_1,122,492). coords(wk2gt_1,115,53,492).  visible(wk2gt_1,492). orientation(wk2gt_2,141,492). coords(wk2gt_2,99,63,492).  visible(wk2gt_2,492).
orientation(wk2gt_1,122,493). coords(wk2gt_1,115,53,493).  visible(wk2gt_1,493). orientation(wk2gt_2,141,493). coords(wk2gt_2,99,63,493).  visible(wk2gt_2,493).
orientation(wk2gt_1,122,494). coords(wk2gt_1,115,53,494).  visible(wk2gt_1,494). orientation(wk2gt_2,141,494). coords(wk2gt_2,99,63,494).  visible(wk2gt_2,494).
orientation(wk2gt_1,122,495). coords(wk2gt_1,115,53,495).  visible(wk2gt_1,495). orientation(wk2gt_2,141,495). coords(wk2gt_2,100,63,495).  visible(wk2gt_2,495).
orientation(wk2gt_1,122,496). coords(wk2gt_1,115,53,496).  visible(wk2gt_1,496). orientation(wk2gt_2,141,496). coords(wk2gt_2,99,63,496).  visible(wk2gt_2,496).
orientation(wk2gt_1,122,497). coords(wk2gt_1,115,53,497).  visible(wk2gt_1,497). orientation(wk2gt_2,141,497). coords(wk2gt_2,99,63,497).  visible(wk2gt_2,497).
orientation(wk2gt_1,122,498). coords(wk2gt_1,115,53,498).  visible(wk2gt_1,498). orientation(wk2gt_2,141,498). coords(wk2gt_2,99,63,498).  visible(wk2gt_2,498).
orientation(wk2gt_1,122,499). coords(wk2gt_1,115,53,499).  visible(wk2gt_1,499). orientation(wk2gt_2,141,499). coords(wk2gt_2,99,63,499).  visible(wk2gt_2,499).
orientation(wk2gt_1,122,500). coords(wk2gt_1,115,53,500).  visible(wk2gt_1,500). orientation(wk2gt_2,141,500). coords(wk2gt_2,99,63,500).  visible(wk2gt_2,500).
orientation(wk2gt_1,122,501). coords(wk2gt_1,115,53,501).  visible(wk2gt_1,501). orientation(wk2gt_2,141,501). coords(wk2gt_2,99,63,501).  visible(wk2gt_2,501).
orientation(wk2gt_1,122,502). coords(wk2gt_1,115,53,502).  visible(wk2gt_1,502). orientation(wk2gt_2,141,502). coords(wk2gt_2,99,63,502).  visible(wk2gt_2,502).
orientation(wk2gt_1,122,503). coords(wk2gt_1,115,53,503).  visible(wk2gt_1,503). orientation(wk2gt_2,141,503). coords(wk2gt_2,99,63,503).  visible(wk2gt_2,503).
orientation(wk2gt_1,122,504). coords(wk2gt_1,115,53,504).  visible(wk2gt_1,504). orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504).
orientation(wk2gt_1,122,505). coords(wk2gt_1,115,53,505).  visible(wk2gt_1,505). orientation(wk2gt_2,141,505). coords(wk2gt_2,99,63,505).  visible(wk2gt_2,505).
orientation(wk2gt_1,122,506). coords(wk2gt_1,115,53,506).  visible(wk2gt_1,506). orientation(wk2gt_2,141,506). coords(wk2gt_2,99,63,506).  visible(wk2gt_2,506).
orientation(wk2gt_1,122,507). coords(wk2gt_1,115,53,507).  visible(wk2gt_1,507). orientation(wk2gt_2,141,507). coords(wk2gt_2,99,63,507).  visible(wk2gt_2,507).
orientation(wk2gt_1,122,508). coords(wk2gt_1,115,53,508).  visible(wk2gt_1,508). orientation(wk2gt_2,141,508). coords(wk2gt_2,99,63,508).  visible(wk2gt_2,508).
orientation(wk2gt_1,122,509). coords(wk2gt_1,115,53,509).  visible(wk2gt_1,509). orientation(wk2gt_2,141,509). coords(wk2gt_2,99,63,509).  visible(wk2gt_2,509).
orientation(wk2gt_1,122,510). coords(wk2gt_1,115,53,510).  visible(wk2gt_1,510). orientation(wk2gt_2,141,510). coords(wk2gt_2,99,63,510).  visible(wk2gt_2,510).
orientation(wk2gt_1,122,511). coords(wk2gt_1,115,53,511).  visible(wk2gt_1,511). orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511).
orientation(wk2gt_1,122,512). coords(wk2gt_1,115,53,512).  visible(wk2gt_1,512). orientation(wk2gt_2,141,512). coords(wk2gt_2,99,63,512).  visible(wk2gt_2,512).
orientation(wk2gt_1,122,513). coords(wk2gt_1,115,53,513).  visible(wk2gt_1,513). orientation(wk2gt_2,141,513). coords(wk2gt_2,99,63,513).  visible(wk2gt_2,513).
orientation(wk2gt_1,122,514). coords(wk2gt_1,115,53,514).  visible(wk2gt_1,514). orientation(wk2gt_2,141,514). coords(wk2gt_2,99,63,514).  visible(wk2gt_2,514).
orientation(wk2gt_1,122,515). coords(wk2gt_1,115,53,515).  visible(wk2gt_1,515). orientation(wk2gt_2,141,515). coords(wk2gt_2,99,63,515).  visible(wk2gt_2,515).
orientation(wk2gt_1,122,516). coords(wk2gt_1,115,53,516).  visible(wk2gt_1,516). orientation(wk2gt_2,141,516). coords(wk2gt_2,99,63,516).  visible(wk2gt_2,516).
orientation(wk2gt_1,122,517). coords(wk2gt_1,115,53,517).  visible(wk2gt_1,517). orientation(wk2gt_2,141,517). coords(wk2gt_2,99,63,517).  visible(wk2gt_2,517).
happensAt(disappear(wk2gt_1),516). happensAt(disappear(wk2gt_2),516).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=492..516.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=492..516.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),492). :- holdsAt(meeting(wk2gt_2,wk2gt_1),492).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),493). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),493).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),494). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),494).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),495). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),495).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),496). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),496).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),497). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),497).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),498). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),498).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),499). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),499).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),500). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),500).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),501). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),501).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),502). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),502).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),503). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),503).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),504). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),504).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),505). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),505).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),506). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),506).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),507). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),507).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),508). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),508).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),509). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),509).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),510). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),510).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),511). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),511).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),512). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),512).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),513). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),513).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),514). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),514).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),515). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),515).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),516). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),516).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),517). :- holdsAt(meeting(wk2gt_2,wk2gt_1),517).
:- simpleEvent(T,P,walking), T>492 , T<516.
:- simpleEvent(T,P,running), T>492 , T<516.

#show simpleEvent/3.