
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

	orientation(br1gt_2,12,514). coords(br1gt_2,86,209,514).  visible(br1gt_2,514). happensAt(inactive(br1gt_2),514). orientation(br1gt_3,157,514). coords(br1gt_3,127,130,514).  visible(br1gt_3,514). happensAt(walking(br1gt_3),514).
orientation(br1gt_2,12,515). coords(br1gt_2,86,209,515).  visible(br1gt_2,515). happensAt(inactive(br1gt_2),515). orientation(br1gt_3,157,515). coords(br1gt_3,128,130,515).  visible(br1gt_3,515). happensAt(walking(br1gt_3),515).
orientation(br1gt_2,12,516). coords(br1gt_2,86,209,516).  visible(br1gt_2,516). happensAt(inactive(br1gt_2),516). orientation(br1gt_3,157,516). coords(br1gt_3,130,130,516).  visible(br1gt_3,516). happensAt(walking(br1gt_3),516).
orientation(br1gt_2,12,517). coords(br1gt_2,86,209,517).  visible(br1gt_2,517). happensAt(inactive(br1gt_2),517). orientation(br1gt_3,157,517). coords(br1gt_3,132,129,517).  visible(br1gt_3,517). happensAt(walking(br1gt_3),517).
orientation(br1gt_2,12,518). coords(br1gt_2,86,209,518).  visible(br1gt_2,518). happensAt(inactive(br1gt_2),518). orientation(br1gt_3,157,518). coords(br1gt_3,133,129,518).  visible(br1gt_3,518). happensAt(walking(br1gt_3),518).
orientation(br1gt_2,12,519). coords(br1gt_2,86,209,519).  visible(br1gt_2,519). happensAt(inactive(br1gt_2),519). orientation(br1gt_3,157,519). coords(br1gt_3,134,129,519).  visible(br1gt_3,519). happensAt(walking(br1gt_3),519).
orientation(br1gt_2,12,520). coords(br1gt_2,86,209,520).  visible(br1gt_2,520). happensAt(inactive(br1gt_2),520). orientation(br1gt_3,155,520). coords(br1gt_3,134,128,520).  visible(br1gt_3,520). happensAt(walking(br1gt_3),520).
orientation(br1gt_2,12,521). coords(br1gt_2,86,209,521).  visible(br1gt_2,521). happensAt(inactive(br1gt_2),521). orientation(br1gt_3,155,521). coords(br1gt_3,135,127,521).  visible(br1gt_3,521). happensAt(walking(br1gt_3),521).
orientation(br1gt_2,12,522). coords(br1gt_2,86,209,522).  visible(br1gt_2,522). happensAt(inactive(br1gt_2),522). orientation(br1gt_3,155,522). coords(br1gt_3,135,127,522).  visible(br1gt_3,522). happensAt(walking(br1gt_3),522).
orientation(br1gt_2,12,523). coords(br1gt_2,86,209,523).  visible(br1gt_2,523). happensAt(inactive(br1gt_2),523). orientation(br1gt_3,155,523). coords(br1gt_3,136,126,523).  visible(br1gt_3,523). happensAt(walking(br1gt_3),523).
orientation(br1gt_2,12,524). coords(br1gt_2,86,209,524).  visible(br1gt_2,524). happensAt(inactive(br1gt_2),524). orientation(br1gt_3,155,524). coords(br1gt_3,136,125,524).  visible(br1gt_3,524). happensAt(walking(br1gt_3),524).
orientation(br1gt_2,12,525). coords(br1gt_2,86,209,525).  visible(br1gt_2,525). happensAt(inactive(br1gt_2),525). orientation(br1gt_3,155,525). coords(br1gt_3,140,124,525).  visible(br1gt_3,525). happensAt(walking(br1gt_3),525).
orientation(br1gt_2,12,526). coords(br1gt_2,86,209,526).  visible(br1gt_2,526). happensAt(inactive(br1gt_2),526). orientation(br1gt_3,155,526). coords(br1gt_3,142,123,526).  visible(br1gt_3,526). happensAt(walking(br1gt_3),526).
orientation(br1gt_2,12,527). coords(br1gt_2,86,209,527).  visible(br1gt_2,527). happensAt(inactive(br1gt_2),527). orientation(br1gt_3,155,527). coords(br1gt_3,145,123,527).  visible(br1gt_3,527). happensAt(walking(br1gt_3),527).
orientation(br1gt_2,12,528). coords(br1gt_2,86,209,528).  visible(br1gt_2,528). happensAt(inactive(br1gt_2),528). orientation(br1gt_3,155,528). coords(br1gt_3,146,122,528).  visible(br1gt_3,528). happensAt(walking(br1gt_3),528).
orientation(br1gt_2,12,529). coords(br1gt_2,86,209,529).  visible(br1gt_2,529). happensAt(inactive(br1gt_2),529). orientation(br1gt_3,155,529). coords(br1gt_3,149,122,529).  visible(br1gt_3,529). happensAt(walking(br1gt_3),529).
orientation(br1gt_2,12,530). coords(br1gt_2,86,209,530).  visible(br1gt_2,530). happensAt(inactive(br1gt_2),530). orientation(br1gt_3,155,530). coords(br1gt_3,150,122,530).  visible(br1gt_3,530). happensAt(walking(br1gt_3),530).
orientation(br1gt_2,12,531). coords(br1gt_2,86,209,531).  visible(br1gt_2,531). happensAt(inactive(br1gt_2),531). orientation(br1gt_3,155,531). coords(br1gt_3,152,121,531).  visible(br1gt_3,531). happensAt(walking(br1gt_3),531).
orientation(br1gt_2,12,532). coords(br1gt_2,86,209,532).  visible(br1gt_2,532). happensAt(inactive(br1gt_2),532). orientation(br1gt_3,155,532). coords(br1gt_3,152,121,532).  visible(br1gt_3,532). happensAt(walking(br1gt_3),532).
orientation(br1gt_2,12,533). coords(br1gt_2,86,209,533).  visible(br1gt_2,533). happensAt(inactive(br1gt_2),533). orientation(br1gt_3,152,533). coords(br1gt_3,153,121,533).  visible(br1gt_3,533). happensAt(walking(br1gt_3),533).
orientation(br1gt_2,12,534). coords(br1gt_2,86,209,534).  visible(br1gt_2,534). happensAt(inactive(br1gt_2),534). orientation(br1gt_3,152,534). coords(br1gt_3,154,120,534).  visible(br1gt_3,534). happensAt(walking(br1gt_3),534).
orientation(br1gt_2,12,535). coords(br1gt_2,86,209,535).  visible(br1gt_2,535). happensAt(inactive(br1gt_2),535). orientation(br1gt_3,152,535). coords(br1gt_3,154,120,535).  visible(br1gt_3,535). happensAt(walking(br1gt_3),535).
orientation(br1gt_2,12,536). coords(br1gt_2,86,209,536).  visible(br1gt_2,536). happensAt(inactive(br1gt_2),536). orientation(br1gt_3,152,536). coords(br1gt_3,155,119,536).  visible(br1gt_3,536). happensAt(walking(br1gt_3),536).
orientation(br1gt_2,12,537). coords(br1gt_2,86,209,537).  visible(br1gt_2,537). happensAt(inactive(br1gt_2),537). orientation(br1gt_3,152,537). coords(br1gt_3,156,119,537).  visible(br1gt_3,537). happensAt(walking(br1gt_3),537).
orientation(br1gt_2,12,538). coords(br1gt_2,86,209,538).  visible(br1gt_2,538). happensAt(inactive(br1gt_2),538). orientation(br1gt_3,152,538). coords(br1gt_3,160,119,538).  visible(br1gt_3,538). happensAt(walking(br1gt_3),538).
#show holdsAt/2.