
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

	orientation(fomdgt3_1,150,517). coords(fomdgt3_1,46,102,517).  visible(fomdgt3_1,517). happensAt(inactive(fomdgt3_1),517). orientation(fomdgt3_7,159,517). coords(fomdgt3_7,141,195,517).  visible(fomdgt3_7,517). happensAt(walking(fomdgt3_7),517).
orientation(fomdgt3_1,150,518). coords(fomdgt3_1,46,102,518).  visible(fomdgt3_1,518). happensAt(inactive(fomdgt3_1),518). orientation(fomdgt3_7,159,518). coords(fomdgt3_7,145,195,518).  visible(fomdgt3_7,518). happensAt(walking(fomdgt3_7),518).
orientation(fomdgt3_1,150,519). coords(fomdgt3_1,46,102,519).  visible(fomdgt3_1,519). happensAt(inactive(fomdgt3_1),519). orientation(fomdgt3_7,159,519). coords(fomdgt3_7,146,195,519).  visible(fomdgt3_7,519). happensAt(walking(fomdgt3_7),519).
orientation(fomdgt3_1,150,520). coords(fomdgt3_1,46,102,520).  visible(fomdgt3_1,520). happensAt(inactive(fomdgt3_1),520). orientation(fomdgt3_7,159,520). coords(fomdgt3_7,146,195,520).  visible(fomdgt3_7,520). happensAt(walking(fomdgt3_7),520).
orientation(fomdgt3_1,150,521). coords(fomdgt3_1,46,102,521).  visible(fomdgt3_1,521). happensAt(inactive(fomdgt3_1),521). orientation(fomdgt3_7,159,521). coords(fomdgt3_7,147,196,521).  visible(fomdgt3_7,521). happensAt(walking(fomdgt3_7),521).
orientation(fomdgt3_1,150,522). coords(fomdgt3_1,46,102,522).  visible(fomdgt3_1,522). happensAt(inactive(fomdgt3_1),522). orientation(fomdgt3_7,159,522). coords(fomdgt3_7,148,196,522).  visible(fomdgt3_7,522). happensAt(walking(fomdgt3_7),522).
orientation(fomdgt3_1,150,523). coords(fomdgt3_1,46,102,523).  visible(fomdgt3_1,523). happensAt(inactive(fomdgt3_1),523). orientation(fomdgt3_7,159,523). coords(fomdgt3_7,149,196,523).  visible(fomdgt3_7,523). happensAt(walking(fomdgt3_7),523).
orientation(fomdgt3_1,150,524). coords(fomdgt3_1,46,102,524).  visible(fomdgt3_1,524). happensAt(inactive(fomdgt3_1),524). orientation(fomdgt3_7,159,524). coords(fomdgt3_7,149,199,524).  visible(fomdgt3_7,524). happensAt(walking(fomdgt3_7),524).
orientation(fomdgt3_1,150,525). coords(fomdgt3_1,46,102,525).  visible(fomdgt3_1,525). happensAt(inactive(fomdgt3_1),525). orientation(fomdgt3_7,159,525). coords(fomdgt3_7,150,199,525).  visible(fomdgt3_7,525). happensAt(walking(fomdgt3_7),525).
orientation(fomdgt3_1,150,526). coords(fomdgt3_1,46,102,526).  visible(fomdgt3_1,526). happensAt(inactive(fomdgt3_1),526). orientation(fomdgt3_7,159,526). coords(fomdgt3_7,150,199,526).  visible(fomdgt3_7,526). happensAt(walking(fomdgt3_7),526).
orientation(fomdgt3_1,150,527). coords(fomdgt3_1,46,102,527).  visible(fomdgt3_1,527). happensAt(inactive(fomdgt3_1),527). orientation(fomdgt3_7,159,527). coords(fomdgt3_7,150,199,527).  visible(fomdgt3_7,527). happensAt(walking(fomdgt3_7),527).
orientation(fomdgt3_1,150,528). coords(fomdgt3_1,46,102,528).  visible(fomdgt3_1,528). happensAt(inactive(fomdgt3_1),528). orientation(fomdgt3_7,159,528). coords(fomdgt3_7,151,200,528).  visible(fomdgt3_7,528). happensAt(walking(fomdgt3_7),528).
orientation(fomdgt3_1,150,529). coords(fomdgt3_1,46,102,529).  visible(fomdgt3_1,529). happensAt(inactive(fomdgt3_1),529). orientation(fomdgt3_7,159,529). coords(fomdgt3_7,155,201,529).  visible(fomdgt3_7,529). happensAt(walking(fomdgt3_7),529).
orientation(fomdgt3_1,150,530). coords(fomdgt3_1,46,102,530).  visible(fomdgt3_1,530). happensAt(inactive(fomdgt3_1),530). orientation(fomdgt3_7,159,530). coords(fomdgt3_7,158,203,530).  visible(fomdgt3_7,530). happensAt(walking(fomdgt3_7),530).
orientation(fomdgt3_1,150,531). coords(fomdgt3_1,46,102,531).  visible(fomdgt3_1,531). happensAt(inactive(fomdgt3_1),531). orientation(fomdgt3_7,159,531). coords(fomdgt3_7,160,205,531).  visible(fomdgt3_7,531). happensAt(walking(fomdgt3_7),531).
orientation(fomdgt3_1,150,532). coords(fomdgt3_1,46,102,532).  visible(fomdgt3_1,532). happensAt(inactive(fomdgt3_1),532). orientation(fomdgt3_7,159,532). coords(fomdgt3_7,163,206,532).  visible(fomdgt3_7,532). happensAt(walking(fomdgt3_7),532).
orientation(fomdgt3_1,150,533). coords(fomdgt3_1,46,102,533).  visible(fomdgt3_1,533). happensAt(inactive(fomdgt3_1),533). orientation(fomdgt3_7,159,533). coords(fomdgt3_7,165,207,533).  visible(fomdgt3_7,533). happensAt(walking(fomdgt3_7),533).
orientation(fomdgt3_1,150,534). coords(fomdgt3_1,46,102,534).  visible(fomdgt3_1,534). happensAt(inactive(fomdgt3_1),534). orientation(fomdgt3_7,159,534). coords(fomdgt3_7,167,207,534).  visible(fomdgt3_7,534). happensAt(walking(fomdgt3_7),534).
orientation(fomdgt3_1,150,535). coords(fomdgt3_1,46,102,535).  visible(fomdgt3_1,535). happensAt(inactive(fomdgt3_1),535). orientation(fomdgt3_7,159,535). coords(fomdgt3_7,168,208,535).  visible(fomdgt3_7,535). happensAt(walking(fomdgt3_7),535).
orientation(fomdgt3_1,150,536). coords(fomdgt3_1,46,102,536).  visible(fomdgt3_1,536). happensAt(inactive(fomdgt3_1),536). orientation(fomdgt3_7,159,536). coords(fomdgt3_7,169,209,536).  visible(fomdgt3_7,536). happensAt(walking(fomdgt3_7),536).
orientation(fomdgt3_1,150,537). coords(fomdgt3_1,46,102,537).  visible(fomdgt3_1,537). happensAt(inactive(fomdgt3_1),537). orientation(fomdgt3_7,159,537). coords(fomdgt3_7,169,209,537).  visible(fomdgt3_7,537). happensAt(walking(fomdgt3_7),537).
orientation(fomdgt3_1,150,538). coords(fomdgt3_1,46,102,538).  visible(fomdgt3_1,538). happensAt(inactive(fomdgt3_1),538). orientation(fomdgt3_7,159,538). coords(fomdgt3_7,169,209,538).  visible(fomdgt3_7,538). happensAt(walking(fomdgt3_7),538).
orientation(fomdgt3_1,150,539). coords(fomdgt3_1,46,102,539).  visible(fomdgt3_1,539). happensAt(inactive(fomdgt3_1),539). orientation(fomdgt3_7,159,539). coords(fomdgt3_7,171,210,539).  visible(fomdgt3_7,539). happensAt(walking(fomdgt3_7),539).
orientation(fomdgt3_1,150,540). coords(fomdgt3_1,46,102,540).  visible(fomdgt3_1,540). happensAt(inactive(fomdgt3_1),540). orientation(fomdgt3_7,159,540). coords(fomdgt3_7,171,212,540).  visible(fomdgt3_7,540). happensAt(walking(fomdgt3_7),540).
orientation(fomdgt3_1,150,541). coords(fomdgt3_1,46,102,541).  happensAt(disappear(fomdgt3_1),541). happensAt(inactive(fomdgt3_1),541). orientation(fomdgt3_7,159,541). coords(fomdgt3_7,171,213,541).  visible(fomdgt3_7,541). happensAt(walking(fomdgt3_7),541).
#show holdsAt/2.