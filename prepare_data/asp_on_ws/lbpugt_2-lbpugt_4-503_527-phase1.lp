
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

	orientation(lbpugt_2,118,503). coords(lbpugt_2,205,37,503).  visible(lbpugt_2,503). happensAt(active(lbpugt_2),503). orientation(lbpugt_4,135,503). coords(lbpugt_4,219,34,503).  happensAt(appear(lbpugt_4),503). happensAt(inactive(lbpugt_4),503).
orientation(lbpugt_2,118,504). coords(lbpugt_2,204,37,504).  visible(lbpugt_2,504). happensAt(walking(lbpugt_2),504). orientation(lbpugt_4,135,504). coords(lbpugt_4,219,34,504).  visible(lbpugt_4,504). happensAt(inactive(lbpugt_4),504).
orientation(lbpugt_2,126,505). coords(lbpugt_2,202,37,505).  visible(lbpugt_2,505). happensAt(walking(lbpugt_2),505). orientation(lbpugt_4,135,505). coords(lbpugt_4,219,34,505).  visible(lbpugt_4,505). happensAt(inactive(lbpugt_4),505).
orientation(lbpugt_2,126,506). coords(lbpugt_2,202,37,506).  visible(lbpugt_2,506). happensAt(walking(lbpugt_2),506). orientation(lbpugt_4,135,506). coords(lbpugt_4,219,34,506).  visible(lbpugt_4,506). happensAt(inactive(lbpugt_4),506).
orientation(lbpugt_2,126,507). coords(lbpugt_2,202,39,507).  visible(lbpugt_2,507). happensAt(walking(lbpugt_2),507). orientation(lbpugt_4,135,507). coords(lbpugt_4,219,34,507).  visible(lbpugt_4,507). happensAt(inactive(lbpugt_4),507).
orientation(lbpugt_2,126,508). coords(lbpugt_2,201,39,508).  visible(lbpugt_2,508). happensAt(walking(lbpugt_2),508). orientation(lbpugt_4,135,508). coords(lbpugt_4,219,34,508).  visible(lbpugt_4,508). happensAt(inactive(lbpugt_4),508).
orientation(lbpugt_2,132,509). coords(lbpugt_2,200,39,509).  visible(lbpugt_2,509). happensAt(walking(lbpugt_2),509). orientation(lbpugt_4,135,509). coords(lbpugt_4,219,34,509).  visible(lbpugt_4,509). happensAt(inactive(lbpugt_4),509).
orientation(lbpugt_2,132,510). coords(lbpugt_2,199,39,510).  visible(lbpugt_2,510). happensAt(walking(lbpugt_2),510). orientation(lbpugt_4,135,510). coords(lbpugt_4,219,34,510).  visible(lbpugt_4,510). happensAt(inactive(lbpugt_4),510).
orientation(lbpugt_2,132,511). coords(lbpugt_2,199,39,511).  visible(lbpugt_2,511). happensAt(walking(lbpugt_2),511). orientation(lbpugt_4,135,511). coords(lbpugt_4,219,34,511).  visible(lbpugt_4,511). happensAt(inactive(lbpugt_4),511).
orientation(lbpugt_2,132,512). coords(lbpugt_2,199,41,512).  visible(lbpugt_2,512). happensAt(walking(lbpugt_2),512). orientation(lbpugt_4,135,512). coords(lbpugt_4,219,34,512).  visible(lbpugt_4,512). happensAt(inactive(lbpugt_4),512).
orientation(lbpugt_2,132,513). coords(lbpugt_2,199,42,513).  visible(lbpugt_2,513). happensAt(walking(lbpugt_2),513). orientation(lbpugt_4,135,513). coords(lbpugt_4,219,34,513).  visible(lbpugt_4,513). happensAt(inactive(lbpugt_4),513).
orientation(lbpugt_2,132,514). coords(lbpugt_2,197,42,514).  visible(lbpugt_2,514). happensAt(walking(lbpugt_2),514). orientation(lbpugt_4,135,514). coords(lbpugt_4,219,34,514).  visible(lbpugt_4,514). happensAt(inactive(lbpugt_4),514).
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
#show holdsAt/2.