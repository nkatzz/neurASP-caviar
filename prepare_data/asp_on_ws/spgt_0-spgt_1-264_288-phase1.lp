
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

	orientation(spgt_0,148,264). coords(spgt_0,132,105,264).  visible(spgt_0,264). happensAt(walking(spgt_0),264). orientation(spgt_1,137,264). coords(spgt_1,123,73,264).  visible(spgt_1,264). happensAt(walking(spgt_1),264).
orientation(spgt_0,145,265). coords(spgt_0,133,105,265).  visible(spgt_0,265). happensAt(walking(spgt_0),265). orientation(spgt_1,137,265). coords(spgt_1,125,73,265).  visible(spgt_1,265). happensAt(walking(spgt_1),265).
orientation(spgt_0,145,266). coords(spgt_0,133,105,266).  visible(spgt_0,266). happensAt(walking(spgt_0),266). orientation(spgt_1,137,266). coords(spgt_1,126,72,266).  visible(spgt_1,266). happensAt(walking(spgt_1),266).
orientation(spgt_0,147,267). coords(spgt_0,135,105,267).  visible(spgt_0,267). happensAt(walking(spgt_0),267). orientation(spgt_1,137,267). coords(spgt_1,126,71,267).  visible(spgt_1,267). happensAt(walking(spgt_1),267).
orientation(spgt_0,147,268). coords(spgt_0,137,106,268).  visible(spgt_0,268). happensAt(walking(spgt_0),268). orientation(spgt_1,137,268). coords(spgt_1,126,71,268).  visible(spgt_1,268). happensAt(walking(spgt_1),268).
orientation(spgt_0,147,269). coords(spgt_0,138,106,269).  visible(spgt_0,269). happensAt(walking(spgt_0),269). orientation(spgt_1,137,269). coords(spgt_1,127,71,269).  visible(spgt_1,269). happensAt(walking(spgt_1),269).
orientation(spgt_0,149,270). coords(spgt_0,139,106,270).  visible(spgt_0,270). happensAt(walking(spgt_0),270). orientation(spgt_1,137,270). coords(spgt_1,127,70,270).  visible(spgt_1,270). happensAt(walking(spgt_1),270).
orientation(spgt_0,149,271). coords(spgt_0,140,106,271).  visible(spgt_0,271). happensAt(walking(spgt_0),271). orientation(spgt_1,137,271). coords(spgt_1,128,69,271).  visible(spgt_1,271). happensAt(walking(spgt_1),271).
orientation(spgt_0,149,272). coords(spgt_0,141,106,272).  visible(spgt_0,272). happensAt(walking(spgt_0),272). orientation(spgt_1,137,272). coords(spgt_1,128,68,272).  visible(spgt_1,272). happensAt(walking(spgt_1),272).
orientation(spgt_0,149,273). coords(spgt_0,142,106,273).  visible(spgt_0,273). happensAt(walking(spgt_0),273). orientation(spgt_1,137,273). coords(spgt_1,128,66,273).  visible(spgt_1,273). happensAt(walking(spgt_1),273).
orientation(spgt_0,150,274). coords(spgt_0,142,106,274).  visible(spgt_0,274). happensAt(walking(spgt_0),274). orientation(spgt_1,137,274). coords(spgt_1,128,65,274).  visible(spgt_1,274). happensAt(walking(spgt_1),274).
orientation(spgt_0,149,275). coords(spgt_0,143,106,275).  visible(spgt_0,275). happensAt(walking(spgt_0),275). orientation(spgt_1,137,275). coords(spgt_1,128,63,275).  visible(spgt_1,275). happensAt(walking(spgt_1),275).
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
#show holdsAt/2.