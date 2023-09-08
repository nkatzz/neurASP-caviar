
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

	orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289). orientation(fcgt_4,166,289). coords(fcgt_4,82,206,289).  visible(fcgt_4,289). happensAt(inactive(fcgt_4),289).
orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290). orientation(fcgt_4,166,290). coords(fcgt_4,82,206,290).  visible(fcgt_4,290). happensAt(inactive(fcgt_4),290).
orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291). orientation(fcgt_4,166,291). coords(fcgt_4,82,206,291).  visible(fcgt_4,291). happensAt(inactive(fcgt_4),291).
orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292). orientation(fcgt_4,166,292). coords(fcgt_4,82,206,292).  visible(fcgt_4,292). happensAt(inactive(fcgt_4),292).
orientation(fcgt_3,164,293). coords(fcgt_3,98,186,293).  visible(fcgt_3,293). happensAt(inactive(fcgt_3),293). orientation(fcgt_4,166,293). coords(fcgt_4,82,206,293).  visible(fcgt_4,293). happensAt(inactive(fcgt_4),293).
orientation(fcgt_3,164,294). coords(fcgt_3,98,186,294).  visible(fcgt_3,294). happensAt(inactive(fcgt_3),294). orientation(fcgt_4,166,294). coords(fcgt_4,82,206,294).  visible(fcgt_4,294). happensAt(inactive(fcgt_4),294).
orientation(fcgt_3,164,295). coords(fcgt_3,98,186,295).  visible(fcgt_3,295). happensAt(inactive(fcgt_3),295). orientation(fcgt_4,166,295). coords(fcgt_4,82,206,295).  visible(fcgt_4,295). happensAt(inactive(fcgt_4),295).
orientation(fcgt_3,164,296). coords(fcgt_3,98,186,296).  visible(fcgt_3,296). happensAt(inactive(fcgt_3),296). orientation(fcgt_4,166,296). coords(fcgt_4,82,206,296).  visible(fcgt_4,296). happensAt(inactive(fcgt_4),296).
orientation(fcgt_3,164,297). coords(fcgt_3,98,187,297).  visible(fcgt_3,297). happensAt(inactive(fcgt_3),297). orientation(fcgt_4,166,297). coords(fcgt_4,82,206,297).  visible(fcgt_4,297). happensAt(inactive(fcgt_4),297).
orientation(fcgt_3,164,298). coords(fcgt_3,98,187,298).  visible(fcgt_3,298). happensAt(inactive(fcgt_3),298). orientation(fcgt_4,166,298). coords(fcgt_4,82,206,298).  visible(fcgt_4,298). happensAt(inactive(fcgt_4),298).
orientation(fcgt_3,164,299). coords(fcgt_3,98,187,299).  visible(fcgt_3,299). happensAt(inactive(fcgt_3),299). orientation(fcgt_4,166,299). coords(fcgt_4,82,206,299).  visible(fcgt_4,299). happensAt(inactive(fcgt_4),299).
orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). happensAt(inactive(fcgt_3),300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300). happensAt(inactive(fcgt_4),300).
orientation(fcgt_3,164,301). coords(fcgt_3,98,187,301).  visible(fcgt_3,301). happensAt(inactive(fcgt_3),301). orientation(fcgt_4,166,301). coords(fcgt_4,82,206,301).  visible(fcgt_4,301). happensAt(inactive(fcgt_4),301).
orientation(fcgt_3,164,302). coords(fcgt_3,98,187,302).  visible(fcgt_3,302). happensAt(inactive(fcgt_3),302). orientation(fcgt_4,166,302). coords(fcgt_4,82,206,302).  visible(fcgt_4,302). happensAt(inactive(fcgt_4),302).
orientation(fcgt_3,164,303). coords(fcgt_3,98,187,303).  visible(fcgt_3,303). happensAt(inactive(fcgt_3),303). orientation(fcgt_4,166,303). coords(fcgt_4,82,206,303).  visible(fcgt_4,303). happensAt(inactive(fcgt_4),303).
orientation(fcgt_3,164,304). coords(fcgt_3,98,187,304).  visible(fcgt_3,304). happensAt(inactive(fcgt_3),304). orientation(fcgt_4,166,304). coords(fcgt_4,82,206,304).  visible(fcgt_4,304). happensAt(inactive(fcgt_4),304).
orientation(fcgt_3,164,305). coords(fcgt_3,98,187,305).  visible(fcgt_3,305). happensAt(inactive(fcgt_3),305). orientation(fcgt_4,166,305). coords(fcgt_4,82,206,305).  visible(fcgt_4,305). happensAt(inactive(fcgt_4),305).
orientation(fcgt_3,164,306). coords(fcgt_3,98,187,306).  visible(fcgt_3,306). happensAt(inactive(fcgt_3),306). orientation(fcgt_4,166,306). coords(fcgt_4,82,206,306).  visible(fcgt_4,306). happensAt(inactive(fcgt_4),306).
orientation(fcgt_3,164,307). coords(fcgt_3,98,187,307).  visible(fcgt_3,307). happensAt(inactive(fcgt_3),307). orientation(fcgt_4,166,307). coords(fcgt_4,82,206,307).  visible(fcgt_4,307). happensAt(inactive(fcgt_4),307).
orientation(fcgt_3,164,308). coords(fcgt_3,98,187,308).  visible(fcgt_3,308). happensAt(inactive(fcgt_3),308). orientation(fcgt_4,166,308). coords(fcgt_4,82,206,308).  visible(fcgt_4,308). happensAt(inactive(fcgt_4),308).
orientation(fcgt_3,164,309). coords(fcgt_3,98,187,309).  visible(fcgt_3,309). happensAt(inactive(fcgt_3),309). orientation(fcgt_4,166,309). coords(fcgt_4,82,206,309).  visible(fcgt_4,309). happensAt(inactive(fcgt_4),309).
orientation(fcgt_3,164,310). coords(fcgt_3,98,187,310).  visible(fcgt_3,310). happensAt(inactive(fcgt_3),310). orientation(fcgt_4,166,310). coords(fcgt_4,82,206,310).  visible(fcgt_4,310). happensAt(inactive(fcgt_4),310).
orientation(fcgt_3,164,311). coords(fcgt_3,98,187,311).  visible(fcgt_3,311). happensAt(inactive(fcgt_3),311). orientation(fcgt_4,166,311). coords(fcgt_4,82,206,311).  visible(fcgt_4,311). happensAt(inactive(fcgt_4),311).
orientation(fcgt_3,164,312). coords(fcgt_3,98,188,312).  visible(fcgt_3,312). happensAt(inactive(fcgt_3),312). orientation(fcgt_4,166,312). coords(fcgt_4,82,206,312).  visible(fcgt_4,312). happensAt(inactive(fcgt_4),312).
#show holdsAt/2.