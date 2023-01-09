
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

	orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_4,166,268). coords(fcgt_4,81,205,268).  visible(fcgt_4,268). happensAt(inactive(fcgt_4),268).
orientation(fcgt_1,151,269). coords(fcgt_1,135,137,269).  visible(fcgt_1,269). happensAt(walking(fcgt_1),269). orientation(fcgt_4,166,269). coords(fcgt_4,81,205,269).  visible(fcgt_4,269). happensAt(inactive(fcgt_4),269).
orientation(fcgt_1,151,270). coords(fcgt_1,133,137,270).  visible(fcgt_1,270). happensAt(walking(fcgt_1),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
orientation(fcgt_1,151,271). coords(fcgt_1,131,136,271).  visible(fcgt_1,271). happensAt(walking(fcgt_1),271). orientation(fcgt_4,166,271). coords(fcgt_4,81,205,271).  visible(fcgt_4,271). happensAt(inactive(fcgt_4),271).
orientation(fcgt_1,151,272). coords(fcgt_1,130,137,272).  visible(fcgt_1,272). happensAt(walking(fcgt_1),272). orientation(fcgt_4,166,272). coords(fcgt_4,81,205,272).  visible(fcgt_4,272). happensAt(inactive(fcgt_4),272).
orientation(fcgt_1,156,273). coords(fcgt_1,127,135,273).  visible(fcgt_1,273). happensAt(walking(fcgt_1),273). orientation(fcgt_4,166,273). coords(fcgt_4,81,205,273).  visible(fcgt_4,273). happensAt(inactive(fcgt_4),273).
orientation(fcgt_1,156,274). coords(fcgt_1,125,133,274).  visible(fcgt_1,274). happensAt(walking(fcgt_1),274). orientation(fcgt_4,166,274). coords(fcgt_4,81,205,274).  visible(fcgt_4,274). happensAt(inactive(fcgt_4),274).
orientation(fcgt_1,156,275). coords(fcgt_1,123,133,275).  visible(fcgt_1,275). happensAt(walking(fcgt_1),275). orientation(fcgt_4,166,275). coords(fcgt_4,81,205,275).  visible(fcgt_4,275). happensAt(inactive(fcgt_4),275).
orientation(fcgt_1,153,276). coords(fcgt_1,122,133,276).  visible(fcgt_1,276). happensAt(walking(fcgt_1),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
orientation(fcgt_1,153,277). coords(fcgt_1,122,133,277).  visible(fcgt_1,277). happensAt(walking(fcgt_1),277). orientation(fcgt_4,166,277). coords(fcgt_4,81,205,277).  visible(fcgt_4,277). happensAt(inactive(fcgt_4),277).
orientation(fcgt_1,153,278). coords(fcgt_1,121,133,278).  visible(fcgt_1,278). happensAt(walking(fcgt_1),278). orientation(fcgt_4,166,278). coords(fcgt_4,81,205,278).  visible(fcgt_4,278). happensAt(inactive(fcgt_4),278).
orientation(fcgt_1,153,279). coords(fcgt_1,121,133,279).  visible(fcgt_1,279). happensAt(walking(fcgt_1),279). orientation(fcgt_4,166,279). coords(fcgt_4,81,205,279).  visible(fcgt_4,279). happensAt(inactive(fcgt_4),279).
orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
orientation(fcgt_1,153,281). coords(fcgt_1,119,129,281).  visible(fcgt_1,281). happensAt(walking(fcgt_1),281). orientation(fcgt_4,166,281). coords(fcgt_4,81,205,281).  visible(fcgt_4,281). happensAt(inactive(fcgt_4),281).
orientation(fcgt_1,153,282). coords(fcgt_1,118,127,282).  visible(fcgt_1,282). happensAt(walking(fcgt_1),282). orientation(fcgt_4,166,282). coords(fcgt_4,81,205,282).  visible(fcgt_4,282). happensAt(inactive(fcgt_4),282).
orientation(fcgt_1,153,283). coords(fcgt_1,115,126,283).  visible(fcgt_1,283). happensAt(walking(fcgt_1),283). orientation(fcgt_4,166,283). coords(fcgt_4,81,205,283).  visible(fcgt_4,283). happensAt(inactive(fcgt_4),283).
orientation(fcgt_1,153,284). coords(fcgt_1,113,126,284).  visible(fcgt_1,284). happensAt(walking(fcgt_1),284). orientation(fcgt_4,166,284). coords(fcgt_4,82,206,284).  visible(fcgt_4,284). happensAt(inactive(fcgt_4),284).
orientation(fcgt_1,153,285). coords(fcgt_1,112,127,285).  visible(fcgt_1,285). happensAt(walking(fcgt_1),285). orientation(fcgt_4,166,285). coords(fcgt_4,82,206,285).  visible(fcgt_4,285). happensAt(inactive(fcgt_4),285).
orientation(fcgt_1,153,286). coords(fcgt_1,111,127,286).  visible(fcgt_1,286). happensAt(walking(fcgt_1),286). orientation(fcgt_4,166,286). coords(fcgt_4,82,206,286).  visible(fcgt_4,286). happensAt(inactive(fcgt_4),286).
orientation(fcgt_1,153,287). coords(fcgt_1,111,127,287).  visible(fcgt_1,287). happensAt(walking(fcgt_1),287). orientation(fcgt_4,166,287). coords(fcgt_4,82,206,287).  visible(fcgt_4,287). happensAt(inactive(fcgt_4),287).
orientation(fcgt_1,153,288). coords(fcgt_1,110,127,288).  visible(fcgt_1,288). happensAt(walking(fcgt_1),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
orientation(fcgt_1,153,289). coords(fcgt_1,110,127,289).  visible(fcgt_1,289). happensAt(walking(fcgt_1),289). orientation(fcgt_4,166,289). coords(fcgt_4,82,206,289).  visible(fcgt_4,289). happensAt(inactive(fcgt_4),289).
orientation(fcgt_1,153,290). coords(fcgt_1,110,128,290).  visible(fcgt_1,290). happensAt(walking(fcgt_1),290). orientation(fcgt_4,166,290). coords(fcgt_4,82,206,290).  visible(fcgt_4,290). happensAt(inactive(fcgt_4),290).
orientation(fcgt_1,153,291). coords(fcgt_1,109,128,291).  visible(fcgt_1,291). happensAt(walking(fcgt_1),291). orientation(fcgt_4,166,291). coords(fcgt_4,82,206,291).  visible(fcgt_4,291). happensAt(inactive(fcgt_4),291).
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). happensAt(walking(fcgt_1),292). orientation(fcgt_4,166,292). coords(fcgt_4,82,206,292).  visible(fcgt_4,292). happensAt(inactive(fcgt_4),292).
#show holdsAt/2.