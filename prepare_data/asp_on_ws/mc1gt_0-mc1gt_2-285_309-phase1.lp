
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

	orientation(mc1gt_0,168,285). coords(mc1gt_0,189,209,285).  visible(mc1gt_0,285). happensAt(walking(mc1gt_0),285). orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285).
orientation(mc1gt_0,168,286). coords(mc1gt_0,189,209,286).  visible(mc1gt_0,286). happensAt(walking(mc1gt_0),286). orientation(mc1gt_2,156,286). coords(mc1gt_2,183,173,286).  visible(mc1gt_2,286). happensAt(walking(mc1gt_2),286).
orientation(mc1gt_0,168,287). coords(mc1gt_0,190,212,287).  visible(mc1gt_0,287). happensAt(walking(mc1gt_0),287). orientation(mc1gt_2,165,287). coords(mc1gt_2,185,176,287).  visible(mc1gt_2,287). happensAt(walking(mc1gt_2),287).
orientation(mc1gt_0,168,288). coords(mc1gt_0,191,213,288).  visible(mc1gt_0,288). happensAt(walking(mc1gt_0),288). orientation(mc1gt_2,163,288). coords(mc1gt_2,187,178,288).  visible(mc1gt_2,288). happensAt(walking(mc1gt_2),288).
orientation(mc1gt_0,168,289). coords(mc1gt_0,191,214,289).  visible(mc1gt_0,289). happensAt(walking(mc1gt_0),289). orientation(mc1gt_2,163,289). coords(mc1gt_2,188,179,289).  visible(mc1gt_2,289). happensAt(walking(mc1gt_2),289).
orientation(mc1gt_0,168,290). coords(mc1gt_0,195,216,290).  visible(mc1gt_0,290). happensAt(walking(mc1gt_0),290). orientation(mc1gt_2,163,290). coords(mc1gt_2,190,180,290).  visible(mc1gt_2,290). happensAt(walking(mc1gt_2),290).
orientation(mc1gt_0,174,291). coords(mc1gt_0,196,217,291).  visible(mc1gt_0,291). happensAt(walking(mc1gt_0),291). orientation(mc1gt_2,166,291). coords(mc1gt_2,191,182,291).  visible(mc1gt_2,291). happensAt(walking(mc1gt_2),291).
orientation(mc1gt_0,179,292). coords(mc1gt_0,197,218,292).  visible(mc1gt_0,292). happensAt(walking(mc1gt_0),292). orientation(mc1gt_2,159,292). coords(mc1gt_2,191,182,292).  visible(mc1gt_2,292). happensAt(walking(mc1gt_2),292).
orientation(mc1gt_0,1,293). coords(mc1gt_0,197,219,293).  visible(mc1gt_0,293). happensAt(walking(mc1gt_0),293). orientation(mc1gt_2,160,293). coords(mc1gt_2,192,183,293).  visible(mc1gt_2,293). happensAt(walking(mc1gt_2),293).
orientation(mc1gt_0,1,294). coords(mc1gt_0,198,220,294).  visible(mc1gt_0,294). happensAt(walking(mc1gt_0),294). orientation(mc1gt_2,165,294). coords(mc1gt_2,192,184,294).  visible(mc1gt_2,294). happensAt(walking(mc1gt_2),294).
orientation(mc1gt_0,1,295). coords(mc1gt_0,199,221,295).  visible(mc1gt_0,295). happensAt(walking(mc1gt_0),295). orientation(mc1gt_2,167,295). coords(mc1gt_2,192,184,295).  visible(mc1gt_2,295). happensAt(walking(mc1gt_2),295).
orientation(mc1gt_0,1,296). coords(mc1gt_0,199,223,296).  visible(mc1gt_0,296). happensAt(walking(mc1gt_0),296). orientation(mc1gt_2,172,296). coords(mc1gt_2,193,185,296).  visible(mc1gt_2,296). happensAt(walking(mc1gt_2),296).
orientation(mc1gt_0,1,297). coords(mc1gt_0,199,224,297).  visible(mc1gt_0,297). happensAt(walking(mc1gt_0),297). orientation(mc1gt_2,168,297). coords(mc1gt_2,194,187,297).  visible(mc1gt_2,297). happensAt(walking(mc1gt_2),297).
orientation(mc1gt_0,1,298). coords(mc1gt_0,199,225,298).  visible(mc1gt_0,298). happensAt(walking(mc1gt_0),298). orientation(mc1gt_2,170,298). coords(mc1gt_2,196,188,298).  visible(mc1gt_2,298). happensAt(walking(mc1gt_2),298).
orientation(mc1gt_0,1,299). coords(mc1gt_0,199,229,299).  visible(mc1gt_0,299). happensAt(walking(mc1gt_0),299). orientation(mc1gt_2,172,299). coords(mc1gt_2,200,190,299).  visible(mc1gt_2,299). happensAt(walking(mc1gt_2),299).
orientation(mc1gt_0,1,300). coords(mc1gt_0,199,231,300).  visible(mc1gt_0,300). happensAt(walking(mc1gt_0),300). orientation(mc1gt_2,170,300). coords(mc1gt_2,202,192,300).  visible(mc1gt_2,300). happensAt(walking(mc1gt_2),300).
orientation(mc1gt_0,1,301). coords(mc1gt_0,200,231,301).  visible(mc1gt_0,301). happensAt(walking(mc1gt_0),301). orientation(mc1gt_2,173,301). coords(mc1gt_2,205,193,301).  visible(mc1gt_2,301). happensAt(walking(mc1gt_2),301).
orientation(mc1gt_0,1,302). coords(mc1gt_0,200,232,302).  visible(mc1gt_0,302). happensAt(walking(mc1gt_0),302). orientation(mc1gt_2,176,302). coords(mc1gt_2,207,195,302).  visible(mc1gt_2,302). happensAt(walking(mc1gt_2),302).
orientation(mc1gt_0,1,303). coords(mc1gt_0,201,233,303).  visible(mc1gt_0,303). happensAt(walking(mc1gt_0),303). orientation(mc1gt_2,178,303). coords(mc1gt_2,208,196,303).  visible(mc1gt_2,303). happensAt(walking(mc1gt_2),303).
orientation(mc1gt_0,1,304). coords(mc1gt_0,202,233,304).  visible(mc1gt_0,304). happensAt(walking(mc1gt_0),304). orientation(mc1gt_2,178,304). coords(mc1gt_2,211,197,304).  visible(mc1gt_2,304). happensAt(walking(mc1gt_2),304).
orientation(mc1gt_0,4,305). coords(mc1gt_0,202,234,305).  visible(mc1gt_0,305). happensAt(walking(mc1gt_0),305). orientation(mc1gt_2,178,305). coords(mc1gt_2,211,199,305).  visible(mc1gt_2,305). happensAt(walking(mc1gt_2),305).
orientation(mc1gt_0,4,306). coords(mc1gt_0,202,234,306).  visible(mc1gt_0,306). happensAt(walking(mc1gt_0),306). orientation(mc1gt_2,178,306). coords(mc1gt_2,212,200,306).  visible(mc1gt_2,306). happensAt(walking(mc1gt_2),306).
orientation(mc1gt_0,4,307). coords(mc1gt_0,201,235,307).  visible(mc1gt_0,307). happensAt(walking(mc1gt_0),307). orientation(mc1gt_2,178,307). coords(mc1gt_2,213,200,307).  visible(mc1gt_2,307). happensAt(walking(mc1gt_2),307).
orientation(mc1gt_0,10,308). coords(mc1gt_0,201,237,308).  visible(mc1gt_0,308). happensAt(walking(mc1gt_0),308). orientation(mc1gt_2,178,308). coords(mc1gt_2,213,202,308).  visible(mc1gt_2,308). happensAt(walking(mc1gt_2),308).
orientation(mc1gt_0,10,309). coords(mc1gt_0,202,238,309).  visible(mc1gt_0,309). happensAt(walking(mc1gt_0),309). orientation(mc1gt_2,178,309). coords(mc1gt_2,214,203,309).  visible(mc1gt_2,309). happensAt(walking(mc1gt_2),309).
#show holdsAt/2.