

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
orientation(mwt2gt_1,164,481). coords(mwt2gt_1,62,138,481).  visible(mwt2gt_1,481). orientation(mwt2gt_2,160,481). coords(mwt2gt_2,59,151,481).  visible(mwt2gt_2,481).
orientation(mwt2gt_1,164,482). coords(mwt2gt_1,61,138,482).  visible(mwt2gt_1,482). orientation(mwt2gt_2,160,482). coords(mwt2gt_2,59,151,482).  visible(mwt2gt_2,482).
orientation(mwt2gt_1,164,483). coords(mwt2gt_1,61,139,483).  visible(mwt2gt_1,483). orientation(mwt2gt_2,147,483). coords(mwt2gt_2,59,153,483).  visible(mwt2gt_2,483).
orientation(mwt2gt_1,156,484). coords(mwt2gt_1,61,139,484).  visible(mwt2gt_1,484). orientation(mwt2gt_2,140,484). coords(mwt2gt_2,58,154,484).  visible(mwt2gt_2,484).
orientation(mwt2gt_1,156,485). coords(mwt2gt_1,60,140,485).  visible(mwt2gt_1,485). orientation(mwt2gt_2,142,485). coords(mwt2gt_2,58,154,485).  visible(mwt2gt_2,485).
orientation(mwt2gt_1,156,486). coords(mwt2gt_1,60,142,486).  visible(mwt2gt_1,486). orientation(mwt2gt_2,147,486). coords(mwt2gt_2,58,154,486).  visible(mwt2gt_2,486).
orientation(mwt2gt_1,156,487). coords(mwt2gt_1,60,144,487).  visible(mwt2gt_1,487). orientation(mwt2gt_2,152,487). coords(mwt2gt_2,58,155,487).  visible(mwt2gt_2,487).
orientation(mwt2gt_1,156,488). coords(mwt2gt_1,60,145,488).  visible(mwt2gt_1,488). orientation(mwt2gt_2,149,488). coords(mwt2gt_2,57,155,488).  visible(mwt2gt_2,488).
orientation(mwt2gt_1,156,489). coords(mwt2gt_1,60,146,489).  visible(mwt2gt_1,489). orientation(mwt2gt_2,146,489). coords(mwt2gt_2,56,155,489).  visible(mwt2gt_2,489).
orientation(mwt2gt_1,156,490). coords(mwt2gt_1,59,146,490).  visible(mwt2gt_1,490). orientation(mwt2gt_2,151,490). coords(mwt2gt_2,56,155,490).  visible(mwt2gt_2,490).
orientation(mwt2gt_1,156,491). coords(mwt2gt_1,59,147,491).  visible(mwt2gt_1,491). orientation(mwt2gt_2,147,491). coords(mwt2gt_2,56,155,491).  visible(mwt2gt_2,491).
orientation(mwt2gt_1,156,492). coords(mwt2gt_1,59,148,492).  visible(mwt2gt_1,492). orientation(mwt2gt_2,145,492). coords(mwt2gt_2,56,157,492).  visible(mwt2gt_2,492).
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
orientation(mwt2gt_1,156,506). coords(mwt2gt_1,54,153,506).  visible(mwt2gt_1,506). orientation(mwt2gt_2,151,506). coords(mwt2gt_2,48,163,506).  visible(mwt2gt_2,506).
happensAt(disappear(mwt2gt_1),505). happensAt(disappear(mwt2gt_2),505).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=481..505.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=481..505.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),481). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),481).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),482). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),482).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),483). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),483).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),484). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),484).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),485). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),485).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),486). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),486).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),487). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),487).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),488). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),488).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),489). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),489).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),490). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),490).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),491). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),491).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),492). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),492).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),493). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),493).
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
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),506). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),506).
:- simpleEvent(T,P,inactive), T>481 , T<505.
:- simpleEvent(T,P,running), T>481 , T<505.

#show simpleEvent/3.