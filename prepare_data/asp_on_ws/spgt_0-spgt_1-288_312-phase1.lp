
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

	orientation(spgt_0,147,288). coords(spgt_0,157,109,288).  visible(spgt_0,288). happensAt(walking(spgt_0),288). orientation(spgt_1,133,288). coords(spgt_1,132,51,288).  visible(spgt_1,288). happensAt(walking(spgt_1),288).
orientation(spgt_0,147,289). coords(spgt_0,158,109,289).  visible(spgt_0,289). happensAt(walking(spgt_0),289). orientation(spgt_1,133,289). coords(spgt_1,133,50,289).  visible(spgt_1,289). happensAt(walking(spgt_1),289).
orientation(spgt_0,147,290). coords(spgt_0,158,110,290).  visible(spgt_0,290). happensAt(walking(spgt_0),290). orientation(spgt_1,133,290). coords(spgt_1,133,49,290).  visible(spgt_1,290). happensAt(walking(spgt_1),290).
orientation(spgt_0,147,291). coords(spgt_0,158,111,291).  visible(spgt_0,291). happensAt(walking(spgt_0),291). orientation(spgt_1,133,291). coords(spgt_1,134,48,291).  visible(spgt_1,291). happensAt(walking(spgt_1),291).
orientation(spgt_0,147,292). coords(spgt_0,159,112,292).  visible(spgt_0,292). happensAt(walking(spgt_0),292). orientation(spgt_1,133,292). coords(spgt_1,134,48,292).  visible(spgt_1,292). happensAt(walking(spgt_1),292).
orientation(spgt_0,147,293). coords(spgt_0,160,114,293).  visible(spgt_0,293). happensAt(walking(spgt_0),293). orientation(spgt_1,133,293). coords(spgt_1,137,48,293).  visible(spgt_1,293). happensAt(walking(spgt_1),293).
orientation(spgt_0,147,294). coords(spgt_0,162,116,294).  visible(spgt_0,294). happensAt(walking(spgt_0),294). orientation(spgt_1,133,294). coords(spgt_1,137,47,294).  visible(spgt_1,294). happensAt(walking(spgt_1),294).
orientation(spgt_0,143,295). coords(spgt_0,164,118,295).  visible(spgt_0,295). happensAt(walking(spgt_0),295). orientation(spgt_1,133,295). coords(spgt_1,137,47,295).  visible(spgt_1,295). happensAt(walking(spgt_1),295).
orientation(spgt_0,145,296). coords(spgt_0,165,119,296).  visible(spgt_0,296). happensAt(walking(spgt_0),296). orientation(spgt_1,133,296). coords(spgt_1,137,48,296).  visible(spgt_1,296). happensAt(walking(spgt_1),296).
orientation(spgt_0,145,297). coords(spgt_0,167,119,297).  visible(spgt_0,297). happensAt(walking(spgt_0),297). orientation(spgt_1,133,297). coords(spgt_1,138,47,297).  visible(spgt_1,297). happensAt(walking(spgt_1),297).
orientation(spgt_0,146,298). coords(spgt_0,168,121,298).  visible(spgt_0,298). happensAt(walking(spgt_0),298). orientation(spgt_1,133,298). coords(spgt_1,138,46,298).  visible(spgt_1,298). happensAt(walking(spgt_1),298).
orientation(spgt_0,147,299). coords(spgt_0,168,121,299).  visible(spgt_0,299). happensAt(walking(spgt_0),299). orientation(spgt_1,133,299). coords(spgt_1,139,46,299).  visible(spgt_1,299). happensAt(walking(spgt_1),299).
orientation(spgt_0,146,300). coords(spgt_0,168,121,300).  visible(spgt_0,300). happensAt(walking(spgt_0),300). orientation(spgt_1,133,300). coords(spgt_1,139,44,300).  visible(spgt_1,300). happensAt(walking(spgt_1),300).
orientation(spgt_0,147,301). coords(spgt_0,169,122,301).  visible(spgt_0,301). happensAt(walking(spgt_0),301). orientation(spgt_1,133,301). coords(spgt_1,139,44,301).  visible(spgt_1,301). happensAt(walking(spgt_1),301).
orientation(spgt_0,147,302). coords(spgt_0,169,123,302).  visible(spgt_0,302). happensAt(walking(spgt_0),302). orientation(spgt_1,133,302). coords(spgt_1,139,44,302).  visible(spgt_1,302). happensAt(walking(spgt_1),302).
orientation(spgt_0,148,303). coords(spgt_0,170,123,303).  visible(spgt_0,303). happensAt(walking(spgt_0),303). orientation(spgt_1,133,303). coords(spgt_1,140,44,303).  visible(spgt_1,303). happensAt(walking(spgt_1),303).
orientation(spgt_0,148,304). coords(spgt_0,171,124,304).  visible(spgt_0,304). happensAt(walking(spgt_0),304). orientation(spgt_1,133,304). coords(spgt_1,140,44,304).  visible(spgt_1,304). happensAt(walking(spgt_1),304).
orientation(spgt_0,149,305). coords(spgt_0,173,124,305).  visible(spgt_0,305). happensAt(walking(spgt_0),305). orientation(spgt_1,133,305). coords(spgt_1,140,44,305).  visible(spgt_1,305). happensAt(walking(spgt_1),305).
orientation(spgt_0,149,306). coords(spgt_0,175,125,306).  visible(spgt_0,306). happensAt(walking(spgt_0),306). orientation(spgt_1,133,306). coords(spgt_1,141,43,306).  visible(spgt_1,306). happensAt(walking(spgt_1),306).
orientation(spgt_0,149,307). coords(spgt_0,178,126,307).  visible(spgt_0,307). happensAt(walking(spgt_0),307). orientation(spgt_1,133,307). coords(spgt_1,142,43,307).  visible(spgt_1,307). happensAt(walking(spgt_1),307).
orientation(spgt_0,149,308). coords(spgt_0,180,128,308).  visible(spgt_0,308). happensAt(walking(spgt_0),308). orientation(spgt_1,133,308). coords(spgt_1,143,43,308).  visible(spgt_1,308). happensAt(walking(spgt_1),308).
orientation(spgt_0,149,309). coords(spgt_0,181,129,309).  visible(spgt_0,309). happensAt(walking(spgt_0),309). orientation(spgt_1,133,309). coords(spgt_1,144,42,309).  visible(spgt_1,309). happensAt(walking(spgt_1),309).
orientation(spgt_0,149,310). coords(spgt_0,182,131,310).  visible(spgt_0,310). happensAt(walking(spgt_0),310). orientation(spgt_1,133,310). coords(spgt_1,144,42,310).  visible(spgt_1,310). happensAt(walking(spgt_1),310).
orientation(spgt_0,149,311). coords(spgt_0,183,133,311).  visible(spgt_0,311). happensAt(walking(spgt_0),311). orientation(spgt_1,133,311). coords(spgt_1,145,42,311).  visible(spgt_1,311). happensAt(walking(spgt_1),311).
orientation(spgt_0,149,312). coords(spgt_0,184,133,312).  visible(spgt_0,312). happensAt(walking(spgt_0),312). orientation(spgt_1,133,312). coords(spgt_1,145,42,312).  visible(spgt_1,312). happensAt(walking(spgt_1),312).
#show holdsAt/2.