

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
orientation(wk1gt_4,147,483). coords(wk1gt_4,56,168,483).  visible(wk1gt_4,483). orientation(wk1gt_5,162,483). coords(wk1gt_5,57,152,483).  visible(wk1gt_5,483).
orientation(wk1gt_4,147,484). coords(wk1gt_4,56,168,484).  visible(wk1gt_4,484). orientation(wk1gt_5,162,484). coords(wk1gt_5,57,152,484).  visible(wk1gt_5,484).
orientation(wk1gt_4,147,485). coords(wk1gt_4,56,169,485).  visible(wk1gt_4,485). orientation(wk1gt_5,162,485). coords(wk1gt_5,57,152,485).  visible(wk1gt_5,485).
orientation(wk1gt_4,147,486). coords(wk1gt_4,56,168,486).  visible(wk1gt_4,486). orientation(wk1gt_5,162,486). coords(wk1gt_5,57,152,486).  visible(wk1gt_5,486).
orientation(wk1gt_4,153,487). coords(wk1gt_4,56,168,487).  visible(wk1gt_4,487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487).
orientation(wk1gt_4,154,488). coords(wk1gt_4,56,168,488).  visible(wk1gt_4,488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488).
orientation(wk1gt_4,154,489). coords(wk1gt_4,56,168,489).  visible(wk1gt_4,489). orientation(wk1gt_5,162,489). coords(wk1gt_5,57,151,489).  visible(wk1gt_5,489).
orientation(wk1gt_4,154,490). coords(wk1gt_4,56,168,490).  visible(wk1gt_4,490). orientation(wk1gt_5,162,490). coords(wk1gt_5,57,150,490).  visible(wk1gt_5,490).
orientation(wk1gt_4,157,491). coords(wk1gt_4,56,169,491).  visible(wk1gt_4,491). orientation(wk1gt_5,162,491). coords(wk1gt_5,57,150,491).  visible(wk1gt_5,491).
orientation(wk1gt_4,157,492). coords(wk1gt_4,56,169,492).  visible(wk1gt_4,492). orientation(wk1gt_5,162,492). coords(wk1gt_5,57,150,492).  visible(wk1gt_5,492).
orientation(wk1gt_4,157,493). coords(wk1gt_4,56,169,493).  visible(wk1gt_4,493). orientation(wk1gt_5,162,493). coords(wk1gt_5,57,150,493).  visible(wk1gt_5,493).
orientation(wk1gt_4,157,494). coords(wk1gt_4,56,170,494).  visible(wk1gt_4,494). orientation(wk1gt_5,162,494). coords(wk1gt_5,57,150,494).  visible(wk1gt_5,494).
orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495).
orientation(wk1gt_4,157,496). coords(wk1gt_4,56,170,496).  visible(wk1gt_4,496). orientation(wk1gt_5,162,496). coords(wk1gt_5,57,150,496).  visible(wk1gt_5,496).
orientation(wk1gt_4,157,497). coords(wk1gt_4,56,170,497).  visible(wk1gt_4,497). orientation(wk1gt_5,162,497). coords(wk1gt_5,57,150,497).  visible(wk1gt_5,497).
orientation(wk1gt_4,157,498). coords(wk1gt_4,56,170,498).  visible(wk1gt_4,498). orientation(wk1gt_5,162,498). coords(wk1gt_5,57,150,498).  visible(wk1gt_5,498).
orientation(wk1gt_4,157,499). coords(wk1gt_4,56,170,499).  visible(wk1gt_4,499). orientation(wk1gt_5,162,499). coords(wk1gt_5,57,150,499).  visible(wk1gt_5,499).
orientation(wk1gt_4,157,500). coords(wk1gt_4,56,170,500).  visible(wk1gt_4,500). orientation(wk1gt_5,162,500). coords(wk1gt_5,57,150,500).  visible(wk1gt_5,500).
orientation(wk1gt_4,157,501). coords(wk1gt_4,56,170,501).  visible(wk1gt_4,501). orientation(wk1gt_5,162,501). coords(wk1gt_5,57,150,501).  visible(wk1gt_5,501).
orientation(wk1gt_4,157,502). coords(wk1gt_4,56,170,502).  visible(wk1gt_4,502). orientation(wk1gt_5,162,502). coords(wk1gt_5,57,150,502).  visible(wk1gt_5,502).
orientation(wk1gt_4,157,503). coords(wk1gt_4,56,170,503).  visible(wk1gt_4,503). orientation(wk1gt_5,162,503). coords(wk1gt_5,57,150,503).  visible(wk1gt_5,503).
orientation(wk1gt_4,157,504). coords(wk1gt_4,56,170,504).  visible(wk1gt_4,504). orientation(wk1gt_5,162,504). coords(wk1gt_5,57,150,504).  visible(wk1gt_5,504).
orientation(wk1gt_4,157,505). coords(wk1gt_4,56,170,505).  visible(wk1gt_4,505). orientation(wk1gt_5,162,505). coords(wk1gt_5,57,150,505).  visible(wk1gt_5,505).
orientation(wk1gt_4,157,506). coords(wk1gt_4,56,170,506).  visible(wk1gt_4,506). orientation(wk1gt_5,162,506). coords(wk1gt_5,57,150,506).  visible(wk1gt_5,506).
orientation(wk1gt_4,157,507). coords(wk1gt_4,56,170,507).  visible(wk1gt_4,507). orientation(wk1gt_5,162,507). coords(wk1gt_5,57,150,507).  visible(wk1gt_5,507).
orientation(wk1gt_4,157,508). coords(wk1gt_4,56,170,508).  visible(wk1gt_4,508). orientation(wk1gt_5,162,508). coords(wk1gt_5,57,150,508).  visible(wk1gt_5,508).
happensAt(disappear(wk1gt_4),507). happensAt(disappear(wk1gt_5),507).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=483..507.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=483..507.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),483). :- holdsAt(meeting(wk1gt_5,wk1gt_4),483).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),484). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),484).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),485). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),485).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),486). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),486).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),487). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),487).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),488). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),488).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),489). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),489).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),490). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),490).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),491). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),491).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),492). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),492).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),493). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),493).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),494). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),494).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),495). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),495).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),496). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),496).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),497). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),497).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),498). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),498).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),499). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),499).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),500). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),500).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),501). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),501).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),502). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),502).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),503). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),503).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),504). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),504).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),505). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),505).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),506). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),506).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),507). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),507).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),508). :- holdsAt(meeting(wk1gt_5,wk1gt_4),508).
:- simpleEvent(T,P,walking), T>498 , T<498.
:- simpleEvent(T,P,running), T>498 , T<498.

#show simpleEvent/3.