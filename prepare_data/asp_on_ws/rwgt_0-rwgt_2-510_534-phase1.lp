
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(rwgt_0,137,510). coords(rwgt_0,316,283,510).  visible(rwgt_0,510). happensAt(inactive(rwgt_0),510). orientation(rwgt_2,124,510). coords(rwgt_2,163,30,510).  visible(rwgt_2,510). happensAt(walking(rwgt_2),510).
orientation(rwgt_0,137,511). coords(rwgt_0,316,283,511).  visible(rwgt_0,511). happensAt(inactive(rwgt_0),511). orientation(rwgt_2,124,511). coords(rwgt_2,162,30,511).  visible(rwgt_2,511). happensAt(walking(rwgt_2),511).
orientation(rwgt_0,137,512). coords(rwgt_0,316,283,512).  visible(rwgt_0,512). happensAt(inactive(rwgt_0),512). orientation(rwgt_2,124,512). coords(rwgt_2,162,30,512).  visible(rwgt_2,512). happensAt(walking(rwgt_2),512).
orientation(rwgt_0,137,513). coords(rwgt_0,316,283,513).  visible(rwgt_0,513). happensAt(inactive(rwgt_0),513). orientation(rwgt_2,124,513). coords(rwgt_2,162,30,513).  visible(rwgt_2,513). happensAt(walking(rwgt_2),513).
orientation(rwgt_0,137,514). coords(rwgt_0,316,283,514).  visible(rwgt_0,514). happensAt(inactive(rwgt_0),514). orientation(rwgt_2,124,514). coords(rwgt_2,162,30,514).  visible(rwgt_2,514). happensAt(walking(rwgt_2),514).
orientation(rwgt_0,137,515). coords(rwgt_0,316,283,515).  visible(rwgt_0,515). happensAt(inactive(rwgt_0),515). orientation(rwgt_2,124,515). coords(rwgt_2,162,30,515).  visible(rwgt_2,515). happensAt(walking(rwgt_2),515).
orientation(rwgt_0,137,516). coords(rwgt_0,316,283,516).  visible(rwgt_0,516). happensAt(inactive(rwgt_0),516). orientation(rwgt_2,129,516). coords(rwgt_2,162,30,516).  visible(rwgt_2,516). happensAt(walking(rwgt_2),516).
orientation(rwgt_0,137,517). coords(rwgt_0,316,283,517).  visible(rwgt_0,517). happensAt(inactive(rwgt_0),517). orientation(rwgt_2,130,517). coords(rwgt_2,162,30,517).  visible(rwgt_2,517). happensAt(walking(rwgt_2),517).
orientation(rwgt_0,137,518). coords(rwgt_0,316,283,518).  visible(rwgt_0,518). happensAt(inactive(rwgt_0),518). orientation(rwgt_2,131,518). coords(rwgt_2,161,30,518).  visible(rwgt_2,518). happensAt(walking(rwgt_2),518).
orientation(rwgt_0,137,519). coords(rwgt_0,316,283,519).  visible(rwgt_0,519). happensAt(inactive(rwgt_0),519). orientation(rwgt_2,131,519). coords(rwgt_2,161,31,519).  visible(rwgt_2,519). happensAt(walking(rwgt_2),519).
orientation(rwgt_0,137,520). coords(rwgt_0,316,283,520).  visible(rwgt_0,520). happensAt(inactive(rwgt_0),520). orientation(rwgt_2,131,520). coords(rwgt_2,160,32,520).  visible(rwgt_2,520). happensAt(walking(rwgt_2),520).
orientation(rwgt_0,137,521). coords(rwgt_0,316,283,521).  visible(rwgt_0,521). happensAt(inactive(rwgt_0),521). orientation(rwgt_2,134,521). coords(rwgt_2,160,33,521).  visible(rwgt_2,521). happensAt(walking(rwgt_2),521).
orientation(rwgt_0,137,522). coords(rwgt_0,316,283,522).  visible(rwgt_0,522). happensAt(inactive(rwgt_0),522). orientation(rwgt_2,136,522). coords(rwgt_2,160,32,522).  visible(rwgt_2,522). happensAt(walking(rwgt_2),522).
orientation(rwgt_0,137,523). coords(rwgt_0,316,283,523).  visible(rwgt_0,523). happensAt(inactive(rwgt_0),523). orientation(rwgt_2,136,523). coords(rwgt_2,159,33,523).  visible(rwgt_2,523). happensAt(walking(rwgt_2),523).
orientation(rwgt_0,137,524). coords(rwgt_0,316,283,524).  visible(rwgt_0,524). happensAt(inactive(rwgt_0),524). orientation(rwgt_2,132,524). coords(rwgt_2,159,33,524).  visible(rwgt_2,524). happensAt(walking(rwgt_2),524).
orientation(rwgt_0,137,525). coords(rwgt_0,316,283,525).  visible(rwgt_0,525). happensAt(inactive(rwgt_0),525). orientation(rwgt_2,132,525). coords(rwgt_2,159,33,525).  visible(rwgt_2,525). happensAt(walking(rwgt_2),525).
orientation(rwgt_0,137,526). coords(rwgt_0,316,283,526).  visible(rwgt_0,526). happensAt(inactive(rwgt_0),526). orientation(rwgt_2,132,526). coords(rwgt_2,159,34,526).  visible(rwgt_2,526). happensAt(walking(rwgt_2),526).
orientation(rwgt_0,137,527). coords(rwgt_0,316,283,527).  visible(rwgt_0,527). happensAt(inactive(rwgt_0),527). orientation(rwgt_2,132,527). coords(rwgt_2,157,34,527).  visible(rwgt_2,527). happensAt(walking(rwgt_2),527).
orientation(rwgt_0,137,528). coords(rwgt_0,316,283,528).  visible(rwgt_0,528). happensAt(inactive(rwgt_0),528). orientation(rwgt_2,132,528). coords(rwgt_2,157,33,528).  visible(rwgt_2,528). happensAt(walking(rwgt_2),528).
orientation(rwgt_0,137,529). coords(rwgt_0,316,283,529).  visible(rwgt_0,529). happensAt(inactive(rwgt_0),529). orientation(rwgt_2,132,529). coords(rwgt_2,157,33,529).  visible(rwgt_2,529). happensAt(walking(rwgt_2),529).
orientation(rwgt_0,137,530). coords(rwgt_0,316,283,530).  visible(rwgt_0,530). happensAt(inactive(rwgt_0),530). orientation(rwgt_2,132,530). coords(rwgt_2,156,34,530).  visible(rwgt_2,530). happensAt(walking(rwgt_2),530).
orientation(rwgt_0,137,531). coords(rwgt_0,316,283,531).  visible(rwgt_0,531). happensAt(inactive(rwgt_0),531). orientation(rwgt_2,132,531). coords(rwgt_2,156,34,531).  visible(rwgt_2,531). happensAt(walking(rwgt_2),531).
orientation(rwgt_0,137,532). coords(rwgt_0,316,283,532).  visible(rwgt_0,532). happensAt(inactive(rwgt_0),532). orientation(rwgt_2,132,532). coords(rwgt_2,155,34,532).  visible(rwgt_2,532). happensAt(walking(rwgt_2),532).
orientation(rwgt_0,137,533). coords(rwgt_0,316,283,533).  visible(rwgt_0,533). happensAt(inactive(rwgt_0),533). orientation(rwgt_2,132,533). coords(rwgt_2,155,35,533).  visible(rwgt_2,533). happensAt(walking(rwgt_2),533).
orientation(rwgt_0,137,534). coords(rwgt_0,316,283,534).  visible(rwgt_0,534). happensAt(inactive(rwgt_0),534). orientation(rwgt_2,132,534). coords(rwgt_2,155,35,534).  visible(rwgt_2,534). happensAt(walking(rwgt_2),534).
#show holdsAt/2.