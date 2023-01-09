
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

	orientation(fomdgt3_2,80,518). coords(fomdgt3_2,237,102,518).  visible(fomdgt3_2,518). happensAt(walking(fomdgt3_2),518). orientation(fomdgt3_7,159,518). coords(fomdgt3_7,145,195,518).  visible(fomdgt3_7,518). happensAt(walking(fomdgt3_7),518).
orientation(fomdgt3_2,80,519). coords(fomdgt3_2,236,102,519).  visible(fomdgt3_2,519). happensAt(walking(fomdgt3_2),519). orientation(fomdgt3_7,159,519). coords(fomdgt3_7,146,195,519).  visible(fomdgt3_7,519). happensAt(walking(fomdgt3_7),519).
orientation(fomdgt3_2,80,520). coords(fomdgt3_2,236,102,520).  visible(fomdgt3_2,520). happensAt(walking(fomdgt3_2),520). orientation(fomdgt3_7,159,520). coords(fomdgt3_7,146,195,520).  visible(fomdgt3_7,520). happensAt(walking(fomdgt3_7),520).
orientation(fomdgt3_2,80,521). coords(fomdgt3_2,235,103,521).  visible(fomdgt3_2,521). happensAt(walking(fomdgt3_2),521). orientation(fomdgt3_7,159,521). coords(fomdgt3_7,147,196,521).  visible(fomdgt3_7,521). happensAt(walking(fomdgt3_7),521).
orientation(fomdgt3_2,80,522). coords(fomdgt3_2,235,103,522).  visible(fomdgt3_2,522). happensAt(walking(fomdgt3_2),522). orientation(fomdgt3_7,159,522). coords(fomdgt3_7,148,196,522).  visible(fomdgt3_7,522). happensAt(walking(fomdgt3_7),522).
orientation(fomdgt3_2,80,523). coords(fomdgt3_2,234,102,523).  visible(fomdgt3_2,523). happensAt(walking(fomdgt3_2),523). orientation(fomdgt3_7,159,523). coords(fomdgt3_7,149,196,523).  visible(fomdgt3_7,523). happensAt(walking(fomdgt3_7),523).
orientation(fomdgt3_2,80,524). coords(fomdgt3_2,234,102,524).  visible(fomdgt3_2,524). happensAt(walking(fomdgt3_2),524). orientation(fomdgt3_7,159,524). coords(fomdgt3_7,149,199,524).  visible(fomdgt3_7,524). happensAt(walking(fomdgt3_7),524).
orientation(fomdgt3_2,80,525). coords(fomdgt3_2,233,102,525).  visible(fomdgt3_2,525). happensAt(walking(fomdgt3_2),525). orientation(fomdgt3_7,159,525). coords(fomdgt3_7,150,199,525).  visible(fomdgt3_7,525). happensAt(walking(fomdgt3_7),525).
orientation(fomdgt3_2,80,526). coords(fomdgt3_2,232,102,526).  visible(fomdgt3_2,526). happensAt(walking(fomdgt3_2),526). orientation(fomdgt3_7,159,526). coords(fomdgt3_7,150,199,526).  visible(fomdgt3_7,526). happensAt(walking(fomdgt3_7),526).
orientation(fomdgt3_2,80,527). coords(fomdgt3_2,231,102,527).  visible(fomdgt3_2,527). happensAt(walking(fomdgt3_2),527). orientation(fomdgt3_7,159,527). coords(fomdgt3_7,150,199,527).  visible(fomdgt3_7,527). happensAt(walking(fomdgt3_7),527).
orientation(fomdgt3_2,80,528). coords(fomdgt3_2,231,101,528).  visible(fomdgt3_2,528). happensAt(walking(fomdgt3_2),528). orientation(fomdgt3_7,159,528). coords(fomdgt3_7,151,200,528).  visible(fomdgt3_7,528). happensAt(walking(fomdgt3_7),528).
orientation(fomdgt3_2,80,529). coords(fomdgt3_2,231,99,529).  visible(fomdgt3_2,529). happensAt(walking(fomdgt3_2),529). orientation(fomdgt3_7,159,529). coords(fomdgt3_7,155,201,529).  visible(fomdgt3_7,529). happensAt(walking(fomdgt3_7),529).
orientation(fomdgt3_2,80,530). coords(fomdgt3_2,231,99,530).  visible(fomdgt3_2,530). happensAt(walking(fomdgt3_2),530). orientation(fomdgt3_7,159,530). coords(fomdgt3_7,158,203,530).  visible(fomdgt3_7,530). happensAt(walking(fomdgt3_7),530).
orientation(fomdgt3_2,80,531). coords(fomdgt3_2,231,98,531).  visible(fomdgt3_2,531). happensAt(walking(fomdgt3_2),531). orientation(fomdgt3_7,159,531). coords(fomdgt3_7,160,205,531).  visible(fomdgt3_7,531). happensAt(walking(fomdgt3_7),531).
orientation(fomdgt3_2,80,532). coords(fomdgt3_2,231,97,532).  visible(fomdgt3_2,532). happensAt(walking(fomdgt3_2),532). orientation(fomdgt3_7,159,532). coords(fomdgt3_7,163,206,532).  visible(fomdgt3_7,532). happensAt(walking(fomdgt3_7),532).
orientation(fomdgt3_2,80,533). coords(fomdgt3_2,231,97,533).  visible(fomdgt3_2,533). happensAt(walking(fomdgt3_2),533). orientation(fomdgt3_7,159,533). coords(fomdgt3_7,165,207,533).  visible(fomdgt3_7,533). happensAt(walking(fomdgt3_7),533).
orientation(fomdgt3_2,80,534). coords(fomdgt3_2,232,97,534).  visible(fomdgt3_2,534). happensAt(walking(fomdgt3_2),534). orientation(fomdgt3_7,159,534). coords(fomdgt3_7,167,207,534).  visible(fomdgt3_7,534). happensAt(walking(fomdgt3_7),534).
orientation(fomdgt3_2,80,535). coords(fomdgt3_2,232,96,535).  visible(fomdgt3_2,535). happensAt(walking(fomdgt3_2),535). orientation(fomdgt3_7,159,535). coords(fomdgt3_7,168,208,535).  visible(fomdgt3_7,535). happensAt(walking(fomdgt3_7),535).
orientation(fomdgt3_2,80,536). coords(fomdgt3_2,232,96,536).  visible(fomdgt3_2,536). happensAt(walking(fomdgt3_2),536). orientation(fomdgt3_7,159,536). coords(fomdgt3_7,169,209,536).  visible(fomdgt3_7,536). happensAt(walking(fomdgt3_7),536).
orientation(fomdgt3_2,80,537). coords(fomdgt3_2,232,96,537).  visible(fomdgt3_2,537). happensAt(walking(fomdgt3_2),537). orientation(fomdgt3_7,159,537). coords(fomdgt3_7,169,209,537).  visible(fomdgt3_7,537). happensAt(walking(fomdgt3_7),537).
orientation(fomdgt3_2,80,538). coords(fomdgt3_2,232,96,538).  visible(fomdgt3_2,538). happensAt(walking(fomdgt3_2),538). orientation(fomdgt3_7,159,538). coords(fomdgt3_7,169,209,538).  visible(fomdgt3_7,538). happensAt(walking(fomdgt3_7),538).
orientation(fomdgt3_2,80,539). coords(fomdgt3_2,232,96,539).  visible(fomdgt3_2,539). happensAt(walking(fomdgt3_2),539). orientation(fomdgt3_7,159,539). coords(fomdgt3_7,171,210,539).  visible(fomdgt3_7,539). happensAt(walking(fomdgt3_7),539).
orientation(fomdgt3_2,80,540). coords(fomdgt3_2,232,96,540).  visible(fomdgt3_2,540). happensAt(walking(fomdgt3_2),540). orientation(fomdgt3_7,159,540). coords(fomdgt3_7,171,212,540).  visible(fomdgt3_7,540). happensAt(walking(fomdgt3_7),540).
orientation(fomdgt3_2,110,541). coords(fomdgt3_2,235,95,541).  visible(fomdgt3_2,541). happensAt(walking(fomdgt3_2),541). orientation(fomdgt3_7,159,541). coords(fomdgt3_7,171,213,541).  visible(fomdgt3_7,541). happensAt(walking(fomdgt3_7),541).
orientation(fomdgt3_2,110,542). coords(fomdgt3_2,236,94,542).  visible(fomdgt3_2,542). happensAt(walking(fomdgt3_2),542). orientation(fomdgt3_7,159,542). coords(fomdgt3_7,172,213,542).  visible(fomdgt3_7,542). happensAt(walking(fomdgt3_7),542).
#show holdsAt/2.