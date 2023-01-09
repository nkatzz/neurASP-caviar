
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

	orientation(br1gt_2,12,502). coords(br1gt_2,86,209,502).  visible(br1gt_2,502). happensAt(inactive(br1gt_2),502). orientation(br1gt_3,157,502). coords(br1gt_3,112,133,502).  visible(br1gt_3,502). happensAt(walking(br1gt_3),502).
orientation(br1gt_2,12,503). coords(br1gt_2,86,209,503).  visible(br1gt_2,503). happensAt(inactive(br1gt_2),503). orientation(br1gt_3,157,503). coords(br1gt_3,113,133,503).  visible(br1gt_3,503). happensAt(walking(br1gt_3),503).
orientation(br1gt_2,12,504). coords(br1gt_2,86,209,504).  visible(br1gt_2,504). happensAt(inactive(br1gt_2),504). orientation(br1gt_3,157,504). coords(br1gt_3,115,133,504).  visible(br1gt_3,504). happensAt(walking(br1gt_3),504).
orientation(br1gt_2,12,505). coords(br1gt_2,86,209,505).  visible(br1gt_2,505). happensAt(inactive(br1gt_2),505). orientation(br1gt_3,157,505). coords(br1gt_3,115,133,505).  visible(br1gt_3,505). happensAt(walking(br1gt_3),505).
orientation(br1gt_2,12,506). coords(br1gt_2,86,209,506).  visible(br1gt_2,506). happensAt(inactive(br1gt_2),506). orientation(br1gt_3,157,506). coords(br1gt_3,116,133,506).  visible(br1gt_3,506). happensAt(walking(br1gt_3),506).
orientation(br1gt_2,12,507). coords(br1gt_2,86,209,507).  visible(br1gt_2,507). happensAt(inactive(br1gt_2),507). orientation(br1gt_3,157,507). coords(br1gt_3,116,132,507).  visible(br1gt_3,507). happensAt(walking(br1gt_3),507).
orientation(br1gt_2,12,508). coords(br1gt_2,86,209,508).  visible(br1gt_2,508). happensAt(inactive(br1gt_2),508). orientation(br1gt_3,157,508). coords(br1gt_3,117,132,508).  visible(br1gt_3,508). happensAt(walking(br1gt_3),508).
orientation(br1gt_2,12,509). coords(br1gt_2,86,209,509).  visible(br1gt_2,509). happensAt(inactive(br1gt_2),509). orientation(br1gt_3,157,509). coords(br1gt_3,117,132,509).  visible(br1gt_3,509). happensAt(walking(br1gt_3),509).
orientation(br1gt_2,12,510). coords(br1gt_2,86,209,510).  visible(br1gt_2,510). happensAt(inactive(br1gt_2),510). orientation(br1gt_3,157,510). coords(br1gt_3,118,132,510).  visible(br1gt_3,510). happensAt(walking(br1gt_3),510).
orientation(br1gt_2,12,511). coords(br1gt_2,86,209,511).  visible(br1gt_2,511). happensAt(inactive(br1gt_2),511). orientation(br1gt_3,155,511). coords(br1gt_3,118,132,511).  visible(br1gt_3,511). happensAt(walking(br1gt_3),511).
orientation(br1gt_2,12,512). coords(br1gt_2,86,209,512).  visible(br1gt_2,512). happensAt(inactive(br1gt_2),512). orientation(br1gt_3,157,512). coords(br1gt_3,122,131,512).  visible(br1gt_3,512). happensAt(walking(br1gt_3),512).
orientation(br1gt_2,12,513). coords(br1gt_2,86,209,513).  visible(br1gt_2,513). happensAt(inactive(br1gt_2),513). orientation(br1gt_3,157,513). coords(br1gt_3,124,131,513).  visible(br1gt_3,513). happensAt(walking(br1gt_3),513).
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
#show holdsAt/2.