

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
orientation(mwt2gt_1,156,493). coords(mwt2gt_1,59,148,493).  visible(mwt2gt_1,493). orientation(mwt2gt_2,146,493). coords(mwt2gt_2,53,158,493).  visible(mwt2gt_2,493).
orientation(mwt2gt_1,156,494). coords(mwt2gt_1,58,148,494).  visible(mwt2gt_1,494). orientation(mwt2gt_2,144,494). coords(mwt2gt_2,53,160,494).  visible(mwt2gt_2,494).
orientation(mwt2gt_1,156,495). coords(mwt2gt_1,57,148,495).  visible(mwt2gt_1,495). orientation(mwt2gt_2,149,495). coords(mwt2gt_2,53,160,495).  visible(mwt2gt_2,495).
orientation(mwt2gt_1,156,496). coords(mwt2gt_1,56,149,496).  visible(mwt2gt_1,496). orientation(mwt2gt_2,151,496). coords(mwt2gt_2,53,161,496).  visible(mwt2gt_2,496).
orientation(mwt2gt_1,156,497). coords(mwt2gt_1,55,149,497).  visible(mwt2gt_1,497). orientation(mwt2gt_2,156,497). coords(mwt2gt_2,53,161,497).  visible(mwt2gt_2,497).
orientation(mwt2gt_1,156,498). coords(mwt2gt_1,55,149,498).  visible(mwt2gt_1,498). orientation(mwt2gt_2,153,498). coords(mwt2gt_2,53,162,498).  visible(mwt2gt_2,498).
orientation(mwt2gt_1,156,499). coords(mwt2gt_1,55,149,499).  visible(mwt2gt_1,499). orientation(mwt2gt_2,151,499). coords(mwt2gt_2,52,162,499).  visible(mwt2gt_2,499).
orientation(mwt2gt_1,156,500). coords(mwt2gt_1,55,150,500).  visible(mwt2gt_1,500). orientation(mwt2gt_2,154,500). coords(mwt2gt_2,52,162,500).  visible(mwt2gt_2,500).
orientation(mwt2gt_1,156,501). coords(mwt2gt_1,55,150,501).  visible(mwt2gt_1,501). orientation(mwt2gt_2,153,501). coords(mwt2gt_2,51,163,501).  visible(mwt2gt_2,501).
orientation(mwt2gt_1,156,502). coords(mwt2gt_1,54,151,502).  visible(mwt2gt_1,502). orientation(mwt2gt_2,153,502). coords(mwt2gt_2,50,163,502).  visible(mwt2gt_2,502).
orientation(mwt2gt_1,156,503). coords(mwt2gt_1,54,152,503).  visible(mwt2gt_1,503). orientation(mwt2gt_2,153,503). coords(mwt2gt_2,49,163,503).  visible(mwt2gt_2,503).
orientation(mwt2gt_1,156,504). coords(mwt2gt_1,54,152,504).  visible(mwt2gt_1,504). orientation(mwt2gt_2,152,504). coords(mwt2gt_2,48,163,504).  visible(mwt2gt_2,504).
orientation(mwt2gt_1,156,505). coords(mwt2gt_1,54,153,505).  visible(mwt2gt_1,505). orientation(mwt2gt_2,151,505). coords(mwt2gt_2,48,163,505).  visible(mwt2gt_2,505).
orientation(mwt2gt_1,156,506). coords(mwt2gt_1,53,153,506).  visible(mwt2gt_1,506). orientation(mwt2gt_2,153,506). coords(mwt2gt_2,48,164,506).  visible(mwt2gt_2,506).
orientation(mwt2gt_1,156,507). coords(mwt2gt_1,53,153,507).  visible(mwt2gt_1,507). orientation(mwt2gt_2,149,507). coords(mwt2gt_2,48,164,507).  visible(mwt2gt_2,507).
orientation(mwt2gt_1,156,508). coords(mwt2gt_1,51,153,508).  visible(mwt2gt_1,508). orientation(mwt2gt_2,156,508). coords(mwt2gt_2,48,164,508).  visible(mwt2gt_2,508).
orientation(mwt2gt_1,156,509). coords(mwt2gt_1,50,154,509).  visible(mwt2gt_1,509). orientation(mwt2gt_2,152,509). coords(mwt2gt_2,48,164,509).  visible(mwt2gt_2,509).
orientation(mwt2gt_1,156,510). coords(mwt2gt_1,50,154,510).  visible(mwt2gt_1,510). orientation(mwt2gt_2,152,510). coords(mwt2gt_2,46,165,510).  visible(mwt2gt_2,510).
orientation(mwt2gt_1,153,511). coords(mwt2gt_1,50,154,511).  visible(mwt2gt_1,511). orientation(mwt2gt_2,153,511). coords(mwt2gt_2,46,165,511).  visible(mwt2gt_2,511).
orientation(mwt2gt_1,153,512). coords(mwt2gt_1,50,155,512).  visible(mwt2gt_1,512). orientation(mwt2gt_2,152,512). coords(mwt2gt_2,46,165,512).  visible(mwt2gt_2,512).
orientation(mwt2gt_1,153,513). coords(mwt2gt_1,50,155,513).  visible(mwt2gt_1,513). orientation(mwt2gt_2,152,513). coords(mwt2gt_2,46,166,513).  visible(mwt2gt_2,513).
orientation(mwt2gt_1,153,514). coords(mwt2gt_1,50,155,514).  visible(mwt2gt_1,514). orientation(mwt2gt_2,152,514). coords(mwt2gt_2,45,166,514).  visible(mwt2gt_2,514).
orientation(mwt2gt_1,153,515). coords(mwt2gt_1,50,155,515).  visible(mwt2gt_1,515). orientation(mwt2gt_2,152,515). coords(mwt2gt_2,45,166,515).  visible(mwt2gt_2,515).
orientation(mwt2gt_1,153,516). coords(mwt2gt_1,50,156,516).  visible(mwt2gt_1,516). orientation(mwt2gt_2,152,516). coords(mwt2gt_2,45,166,516).  visible(mwt2gt_2,516).
orientation(mwt2gt_1,153,517). coords(mwt2gt_1,50,155,517).  visible(mwt2gt_1,517). orientation(mwt2gt_2,152,517). coords(mwt2gt_2,44,166,517).  visible(mwt2gt_2,517).
orientation(mwt2gt_1,153,518). coords(mwt2gt_1,50,155,518).  visible(mwt2gt_1,518). orientation(mwt2gt_2,152,518). coords(mwt2gt_2,44,166,518).  visible(mwt2gt_2,518).
happensAt(disappear(mwt2gt_1),517). happensAt(disappear(mwt2gt_2),517).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=493..517.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=493..517.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),493). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),493).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),494). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),494).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),495). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),495).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),496). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),496).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),497). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),497).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),498). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),498).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),499). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),499).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),500). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),500).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),501). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),501).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),502). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),502).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),503). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),503).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),504). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),504).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),505). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),505).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),506). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),506).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),507). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),507).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),508). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),508).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),509). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),509).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),510). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),510).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),511). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),511).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),512). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),512).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),513). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),513).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),514). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),514).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),515). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),515).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),516). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),516).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),517). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),517).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),518). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),518).
:- simpleEvent(T,P,inactive), T>493 , T<517.
:- simpleEvent(T,P,running), T>493 , T<517.

#show simpleEvent/3.