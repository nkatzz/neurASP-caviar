
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

	orientation(lbpugt_2,132,515). coords(lbpugt_2,196,43,515).  visible(lbpugt_2,515). happensAt(walking(lbpugt_2),515). orientation(lbpugt_4,135,515). coords(lbpugt_4,219,34,515).  visible(lbpugt_4,515). happensAt(inactive(lbpugt_4),515).
orientation(lbpugt_2,132,516). coords(lbpugt_2,196,43,516).  visible(lbpugt_2,516). happensAt(walking(lbpugt_2),516). orientation(lbpugt_4,135,516). coords(lbpugt_4,219,34,516).  visible(lbpugt_4,516). happensAt(inactive(lbpugt_4),516).
orientation(lbpugt_2,132,517). coords(lbpugt_2,196,43,517).  visible(lbpugt_2,517). happensAt(walking(lbpugt_2),517). orientation(lbpugt_4,135,517). coords(lbpugt_4,219,34,517).  visible(lbpugt_4,517). happensAt(inactive(lbpugt_4),517).
orientation(lbpugt_2,132,518). coords(lbpugt_2,195,44,518).  visible(lbpugt_2,518). happensAt(walking(lbpugt_2),518). orientation(lbpugt_4,135,518). coords(lbpugt_4,219,34,518).  visible(lbpugt_4,518). happensAt(inactive(lbpugt_4),518).
orientation(lbpugt_2,132,519). coords(lbpugt_2,194,44,519).  visible(lbpugt_2,519). happensAt(walking(lbpugt_2),519). orientation(lbpugt_4,135,519). coords(lbpugt_4,219,34,519).  visible(lbpugt_4,519). happensAt(inactive(lbpugt_4),519).
orientation(lbpugt_2,132,520). coords(lbpugt_2,194,44,520).  visible(lbpugt_2,520). happensAt(walking(lbpugt_2),520). orientation(lbpugt_4,135,520). coords(lbpugt_4,219,34,520).  visible(lbpugt_4,520). happensAt(inactive(lbpugt_4),520).
orientation(lbpugt_2,132,521). coords(lbpugt_2,193,44,521).  visible(lbpugt_2,521). happensAt(walking(lbpugt_2),521). orientation(lbpugt_4,135,521). coords(lbpugt_4,219,34,521).  visible(lbpugt_4,521). happensAt(inactive(lbpugt_4),521).
orientation(lbpugt_2,132,522). coords(lbpugt_2,193,44,522).  visible(lbpugt_2,522). happensAt(walking(lbpugt_2),522). orientation(lbpugt_4,135,522). coords(lbpugt_4,219,34,522).  visible(lbpugt_4,522). happensAt(inactive(lbpugt_4),522).
orientation(lbpugt_2,132,523). coords(lbpugt_2,193,44,523).  visible(lbpugt_2,523). happensAt(walking(lbpugt_2),523). orientation(lbpugt_4,135,523). coords(lbpugt_4,219,34,523).  visible(lbpugt_4,523). happensAt(inactive(lbpugt_4),523).
orientation(lbpugt_2,132,524). coords(lbpugt_2,193,44,524).  visible(lbpugt_2,524). happensAt(walking(lbpugt_2),524). orientation(lbpugt_4,135,524). coords(lbpugt_4,219,34,524).  visible(lbpugt_4,524). happensAt(inactive(lbpugt_4),524).
orientation(lbpugt_2,132,525). coords(lbpugt_2,193,44,525).  visible(lbpugt_2,525). happensAt(walking(lbpugt_2),525). orientation(lbpugt_4,135,525). coords(lbpugt_4,219,34,525).  visible(lbpugt_4,525). happensAt(inactive(lbpugt_4),525).
orientation(lbpugt_2,132,526). coords(lbpugt_2,193,44,526).  visible(lbpugt_2,526). happensAt(walking(lbpugt_2),526). orientation(lbpugt_4,135,526). coords(lbpugt_4,219,34,526).  visible(lbpugt_4,526). happensAt(inactive(lbpugt_4),526).
orientation(lbpugt_2,132,527). coords(lbpugt_2,193,44,527).  visible(lbpugt_2,527). happensAt(walking(lbpugt_2),527). orientation(lbpugt_4,135,527). coords(lbpugt_4,219,34,527).  visible(lbpugt_4,527). happensAt(inactive(lbpugt_4),527).
orientation(lbpugt_2,132,528). coords(lbpugt_2,192,44,528).  visible(lbpugt_2,528). happensAt(walking(lbpugt_2),528). orientation(lbpugt_4,135,528). coords(lbpugt_4,219,34,528).  visible(lbpugt_4,528). happensAt(inactive(lbpugt_4),528).
orientation(lbpugt_2,132,529). coords(lbpugt_2,192,44,529).  visible(lbpugt_2,529). happensAt(walking(lbpugt_2),529). orientation(lbpugt_4,135,529). coords(lbpugt_4,219,34,529).  visible(lbpugt_4,529). happensAt(inactive(lbpugt_4),529).
orientation(lbpugt_2,132,530). coords(lbpugt_2,192,44,530).  visible(lbpugt_2,530). happensAt(walking(lbpugt_2),530). orientation(lbpugt_4,135,530). coords(lbpugt_4,219,34,530).  visible(lbpugt_4,530). happensAt(inactive(lbpugt_4),530).
orientation(lbpugt_2,132,531). coords(lbpugt_2,191,44,531).  visible(lbpugt_2,531). happensAt(walking(lbpugt_2),531). orientation(lbpugt_4,135,531). coords(lbpugt_4,219,34,531).  visible(lbpugt_4,531). happensAt(inactive(lbpugt_4),531).
orientation(lbpugt_2,132,532). coords(lbpugt_2,190,44,532).  visible(lbpugt_2,532). happensAt(walking(lbpugt_2),532). orientation(lbpugt_4,135,532). coords(lbpugt_4,219,34,532).  visible(lbpugt_4,532). happensAt(inactive(lbpugt_4),532).
orientation(lbpugt_2,132,533). coords(lbpugt_2,189,44,533).  visible(lbpugt_2,533). happensAt(walking(lbpugt_2),533). orientation(lbpugt_4,135,533). coords(lbpugt_4,219,34,533).  visible(lbpugt_4,533). happensAt(inactive(lbpugt_4),533).
orientation(lbpugt_2,132,534). coords(lbpugt_2,188,44,534).  visible(lbpugt_2,534). happensAt(walking(lbpugt_2),534). orientation(lbpugt_4,135,534). coords(lbpugt_4,219,34,534).  visible(lbpugt_4,534). happensAt(inactive(lbpugt_4),534).
orientation(lbpugt_2,132,535). coords(lbpugt_2,187,44,535).  visible(lbpugt_2,535). happensAt(walking(lbpugt_2),535). orientation(lbpugt_4,135,535). coords(lbpugt_4,219,34,535).  visible(lbpugt_4,535). happensAt(inactive(lbpugt_4),535).
orientation(lbpugt_2,132,536). coords(lbpugt_2,186,44,536).  visible(lbpugt_2,536). happensAt(walking(lbpugt_2),536). orientation(lbpugt_4,135,536). coords(lbpugt_4,219,34,536).  visible(lbpugt_4,536). happensAt(inactive(lbpugt_4),536).
orientation(lbpugt_2,132,537). coords(lbpugt_2,186,45,537).  visible(lbpugt_2,537). happensAt(walking(lbpugt_2),537). orientation(lbpugt_4,135,537). coords(lbpugt_4,219,34,537).  visible(lbpugt_4,537). happensAt(inactive(lbpugt_4),537).
orientation(lbpugt_2,132,538). coords(lbpugt_2,184,45,538).  visible(lbpugt_2,538). happensAt(walking(lbpugt_2),538). orientation(lbpugt_4,135,538). coords(lbpugt_4,219,34,538).  visible(lbpugt_4,538). happensAt(inactive(lbpugt_4),538).
orientation(lbpugt_2,132,539). coords(lbpugt_2,183,47,539).  visible(lbpugt_2,539). happensAt(walking(lbpugt_2),539). orientation(lbpugt_4,135,539). coords(lbpugt_4,219,34,539).  visible(lbpugt_4,539). happensAt(inactive(lbpugt_4),539).
#show holdsAt/2.