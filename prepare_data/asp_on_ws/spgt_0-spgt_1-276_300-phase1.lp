
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

	orientation(spgt_0,147,276). coords(spgt_0,143,106,276).  visible(spgt_0,276). happensAt(walking(spgt_0),276). orientation(spgt_1,137,276). coords(spgt_1,129,62,276).  visible(spgt_1,276). happensAt(walking(spgt_1),276).
orientation(spgt_0,147,277). coords(spgt_0,143,107,277).  visible(spgt_0,277). happensAt(walking(spgt_0),277). orientation(spgt_1,135,277). coords(spgt_1,129,60,277).  visible(spgt_1,277). happensAt(walking(spgt_1),277).
orientation(spgt_0,147,278). coords(spgt_0,144,107,278).  visible(spgt_0,278). happensAt(walking(spgt_0),278). orientation(spgt_1,134,278). coords(spgt_1,129,60,278).  visible(spgt_1,278). happensAt(walking(spgt_1),278).
orientation(spgt_0,147,279). coords(spgt_0,145,107,279).  visible(spgt_0,279). happensAt(walking(spgt_0),279). orientation(spgt_1,134,279). coords(spgt_1,129,59,279).  visible(spgt_1,279). happensAt(walking(spgt_1),279).
orientation(spgt_0,147,280). coords(spgt_0,148,107,280).  visible(spgt_0,280). happensAt(walking(spgt_0),280). orientation(spgt_1,133,280). coords(spgt_1,129,59,280).  visible(spgt_1,280). happensAt(walking(spgt_1),280).
orientation(spgt_0,147,281). coords(spgt_0,150,107,281).  visible(spgt_0,281). happensAt(walking(spgt_0),281). orientation(spgt_1,131,281). coords(spgt_1,129,58,281).  visible(spgt_1,281). happensAt(walking(spgt_1),281).
orientation(spgt_0,147,282). coords(spgt_0,152,107,282).  visible(spgt_0,282). happensAt(walking(spgt_0),282). orientation(spgt_1,132,282). coords(spgt_1,130,58,282).  visible(spgt_1,282). happensAt(walking(spgt_1),282).
orientation(spgt_0,147,283). coords(spgt_0,153,107,283).  visible(spgt_0,283). happensAt(walking(spgt_0),283). orientation(spgt_1,132,283). coords(spgt_1,130,57,283).  visible(spgt_1,283). happensAt(walking(spgt_1),283).
orientation(spgt_0,147,284). coords(spgt_0,154,107,284).  visible(spgt_0,284). happensAt(walking(spgt_0),284). orientation(spgt_1,133,284). coords(spgt_1,132,57,284).  visible(spgt_1,284). happensAt(walking(spgt_1),284).
orientation(spgt_0,147,285). coords(spgt_0,155,108,285).  visible(spgt_0,285). happensAt(walking(spgt_0),285). orientation(spgt_1,133,285). coords(spgt_1,132,56,285).  visible(spgt_1,285). happensAt(walking(spgt_1),285).
orientation(spgt_0,147,286). coords(spgt_0,157,109,286).  visible(spgt_0,286). happensAt(walking(spgt_0),286). orientation(spgt_1,133,286). coords(spgt_1,132,54,286).  visible(spgt_1,286). happensAt(walking(spgt_1),286).
orientation(spgt_0,147,287). coords(spgt_0,157,109,287).  visible(spgt_0,287). happensAt(walking(spgt_0),287). orientation(spgt_1,133,287). coords(spgt_1,132,53,287).  visible(spgt_1,287). happensAt(walking(spgt_1),287).
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
#show holdsAt/2.