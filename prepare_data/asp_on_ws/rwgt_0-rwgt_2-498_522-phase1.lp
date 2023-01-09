
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

	orientation(rwgt_0,137,498). coords(rwgt_0,316,283,498).  visible(rwgt_0,498). happensAt(inactive(rwgt_0),498). orientation(rwgt_2,124,498). coords(rwgt_2,166,27,498).  visible(rwgt_2,498). happensAt(walking(rwgt_2),498).
orientation(rwgt_0,137,499). coords(rwgt_0,316,283,499).  visible(rwgt_0,499). happensAt(inactive(rwgt_0),499). orientation(rwgt_2,124,499). coords(rwgt_2,166,28,499).  visible(rwgt_2,499). happensAt(walking(rwgt_2),499).
orientation(rwgt_0,137,500). coords(rwgt_0,316,283,500).  visible(rwgt_0,500). happensAt(inactive(rwgt_0),500). orientation(rwgt_2,124,500). coords(rwgt_2,165,29,500).  visible(rwgt_2,500). happensAt(walking(rwgt_2),500).
orientation(rwgt_0,137,501). coords(rwgt_0,316,283,501).  visible(rwgt_0,501). happensAt(inactive(rwgt_0),501). orientation(rwgt_2,124,501). coords(rwgt_2,165,29,501).  visible(rwgt_2,501). happensAt(walking(rwgt_2),501).
orientation(rwgt_0,137,502). coords(rwgt_0,316,283,502).  visible(rwgt_0,502). happensAt(inactive(rwgt_0),502). orientation(rwgt_2,124,502). coords(rwgt_2,165,29,502).  visible(rwgt_2,502). happensAt(walking(rwgt_2),502).
orientation(rwgt_0,137,503). coords(rwgt_0,316,283,503).  visible(rwgt_0,503). happensAt(inactive(rwgt_0),503). orientation(rwgt_2,124,503). coords(rwgt_2,165,29,503).  visible(rwgt_2,503). happensAt(walking(rwgt_2),503).
orientation(rwgt_0,137,504). coords(rwgt_0,316,283,504).  visible(rwgt_0,504). happensAt(inactive(rwgt_0),504). orientation(rwgt_2,124,504). coords(rwgt_2,164,30,504).  visible(rwgt_2,504). happensAt(walking(rwgt_2),504).
orientation(rwgt_0,137,505). coords(rwgt_0,316,283,505).  visible(rwgt_0,505). happensAt(inactive(rwgt_0),505). orientation(rwgt_2,124,505). coords(rwgt_2,164,30,505).  visible(rwgt_2,505). happensAt(walking(rwgt_2),505).
orientation(rwgt_0,137,506). coords(rwgt_0,316,283,506).  visible(rwgt_0,506). happensAt(inactive(rwgt_0),506). orientation(rwgt_2,124,506). coords(rwgt_2,164,30,506).  visible(rwgt_2,506). happensAt(walking(rwgt_2),506).
orientation(rwgt_0,137,507). coords(rwgt_0,316,283,507).  visible(rwgt_0,507). happensAt(inactive(rwgt_0),507). orientation(rwgt_2,124,507). coords(rwgt_2,164,30,507).  visible(rwgt_2,507). happensAt(walking(rwgt_2),507).
orientation(rwgt_0,137,508). coords(rwgt_0,316,283,508).  visible(rwgt_0,508). happensAt(inactive(rwgt_0),508). orientation(rwgt_2,124,508). coords(rwgt_2,164,30,508).  visible(rwgt_2,508). happensAt(walking(rwgt_2),508).
orientation(rwgt_0,137,509). coords(rwgt_0,316,283,509).  visible(rwgt_0,509). happensAt(inactive(rwgt_0),509). orientation(rwgt_2,124,509). coords(rwgt_2,163,30,509).  visible(rwgt_2,509). happensAt(walking(rwgt_2),509).
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
#show holdsAt/2.