
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

	orientation(rwgt_0,2,75). coords(rwgt_0,270,239,75).  visible(rwgt_0,75). happensAt(walking(rwgt_0),75). orientation(rwgt_1,154,75). coords(rwgt_1,208,203,75).  visible(rwgt_1,75). happensAt(walking(rwgt_1),75).
orientation(rwgt_0,2,76). coords(rwgt_0,270,239,76).  visible(rwgt_0,76). happensAt(walking(rwgt_0),76). orientation(rwgt_1,157,76). coords(rwgt_1,206,200,76).  visible(rwgt_1,76). happensAt(walking(rwgt_1),76).
orientation(rwgt_0,2,77). coords(rwgt_0,269,237,77).  visible(rwgt_0,77). happensAt(walking(rwgt_0),77). orientation(rwgt_1,149,77). coords(rwgt_1,203,196,77).  visible(rwgt_1,77). happensAt(walking(rwgt_1),77).
orientation(rwgt_0,166,78). coords(rwgt_0,268,236,78).  visible(rwgt_0,78). happensAt(walking(rwgt_0),78). orientation(rwgt_1,149,78). coords(rwgt_1,201,193,78).  visible(rwgt_1,78). happensAt(walking(rwgt_1),78).
orientation(rwgt_0,166,79). coords(rwgt_0,267,234,79).  visible(rwgt_0,79). happensAt(walking(rwgt_0),79). orientation(rwgt_1,149,79). coords(rwgt_1,201,189,79).  visible(rwgt_1,79). happensAt(walking(rwgt_1),79).
orientation(rwgt_0,160,80). coords(rwgt_0,266,232,80).  visible(rwgt_0,80). happensAt(walking(rwgt_0),80). orientation(rwgt_1,149,80). coords(rwgt_1,200,188,80).  visible(rwgt_1,80). happensAt(walking(rwgt_1),80).
orientation(rwgt_0,160,81). coords(rwgt_0,265,230,81).  visible(rwgt_0,81). happensAt(walking(rwgt_0),81). orientation(rwgt_1,149,81). coords(rwgt_1,199,186,81).  visible(rwgt_1,81). happensAt(walking(rwgt_1),81).
orientation(rwgt_0,160,82). coords(rwgt_0,264,228,82).  visible(rwgt_0,82). happensAt(walking(rwgt_0),82). orientation(rwgt_1,149,82). coords(rwgt_1,198,185,82).  visible(rwgt_1,82). happensAt(walking(rwgt_1),82).
orientation(rwgt_0,160,83). coords(rwgt_0,263,226,83).  visible(rwgt_0,83). happensAt(walking(rwgt_0),83). orientation(rwgt_1,149,83). coords(rwgt_1,198,183,83).  visible(rwgt_1,83). happensAt(walking(rwgt_1),83).
orientation(rwgt_0,160,84). coords(rwgt_0,261,225,84).  visible(rwgt_0,84). happensAt(walking(rwgt_0),84). orientation(rwgt_1,149,84). coords(rwgt_1,197,182,84).  visible(rwgt_1,84). happensAt(walking(rwgt_1),84).
orientation(rwgt_0,160,85). coords(rwgt_0,260,223,85).  visible(rwgt_0,85). happensAt(walking(rwgt_0),85). orientation(rwgt_1,149,85). coords(rwgt_1,196,181,85).  visible(rwgt_1,85). happensAt(walking(rwgt_1),85).
orientation(rwgt_0,160,86). coords(rwgt_0,259,218,86).  visible(rwgt_0,86). happensAt(walking(rwgt_0),86). orientation(rwgt_1,149,86). coords(rwgt_1,195,179,86).  visible(rwgt_1,86). happensAt(walking(rwgt_1),86).
orientation(rwgt_0,160,87). coords(rwgt_0,255,216,87).  visible(rwgt_0,87). happensAt(walking(rwgt_0),87). orientation(rwgt_1,149,87). coords(rwgt_1,194,177,87).  visible(rwgt_1,87). happensAt(walking(rwgt_1),87).
orientation(rwgt_0,168,88). coords(rwgt_0,252,212,88).  visible(rwgt_0,88). happensAt(walking(rwgt_0),88). orientation(rwgt_1,149,88). coords(rwgt_1,193,173,88).  visible(rwgt_1,88). happensAt(walking(rwgt_1),88).
orientation(rwgt_0,168,89). coords(rwgt_0,249,208,89).  visible(rwgt_0,89). happensAt(walking(rwgt_0),89). orientation(rwgt_1,149,89). coords(rwgt_1,190,170,89).  visible(rwgt_1,89). happensAt(walking(rwgt_1),89).
orientation(rwgt_0,168,90). coords(rwgt_0,246,206,90).  visible(rwgt_0,90). happensAt(walking(rwgt_0),90). orientation(rwgt_1,149,90). coords(rwgt_1,188,167,90).  visible(rwgt_1,90). happensAt(walking(rwgt_1),90).
orientation(rwgt_0,168,91). coords(rwgt_0,243,205,91).  visible(rwgt_0,91). happensAt(walking(rwgt_0),91). orientation(rwgt_1,149,91). coords(rwgt_1,185,164,91).  visible(rwgt_1,91). happensAt(walking(rwgt_1),91).
orientation(rwgt_0,168,92). coords(rwgt_0,243,203,92).  visible(rwgt_0,92). happensAt(walking(rwgt_0),92). orientation(rwgt_1,149,92). coords(rwgt_1,184,162,92).  visible(rwgt_1,92). happensAt(walking(rwgt_1),92).
orientation(rwgt_0,168,93). coords(rwgt_0,242,201,93).  visible(rwgt_0,93). happensAt(walking(rwgt_0),93). orientation(rwgt_1,149,93). coords(rwgt_1,182,161,93).  visible(rwgt_1,93). happensAt(walking(rwgt_1),93).
orientation(rwgt_0,168,94). coords(rwgt_0,241,200,94).  visible(rwgt_0,94). happensAt(walking(rwgt_0),94). orientation(rwgt_1,149,94). coords(rwgt_1,182,160,94).  visible(rwgt_1,94). happensAt(walking(rwgt_1),94).
orientation(rwgt_0,168,95). coords(rwgt_0,241,198,95).  visible(rwgt_0,95). happensAt(walking(rwgt_0),95). orientation(rwgt_1,149,95). coords(rwgt_1,181,159,95).  visible(rwgt_1,95). happensAt(walking(rwgt_1),95).
orientation(rwgt_0,168,96). coords(rwgt_0,240,197,96).  visible(rwgt_0,96). happensAt(walking(rwgt_0),96). orientation(rwgt_1,149,96). coords(rwgt_1,181,158,96).  visible(rwgt_1,96). happensAt(walking(rwgt_1),96).
orientation(rwgt_0,153,97). coords(rwgt_0,239,196,97).  visible(rwgt_0,97). happensAt(walking(rwgt_0),97). orientation(rwgt_1,149,97). coords(rwgt_1,181,156,97).  visible(rwgt_1,97). happensAt(walking(rwgt_1),97).
orientation(rwgt_0,152,98). coords(rwgt_0,237,196,98).  visible(rwgt_0,98). happensAt(walking(rwgt_0),98). orientation(rwgt_1,149,98). coords(rwgt_1,180,155,98).  visible(rwgt_1,98). happensAt(walking(rwgt_1),98).
orientation(rwgt_0,148,99). coords(rwgt_0,235,195,99).  visible(rwgt_0,99). happensAt(walking(rwgt_0),99). orientation(rwgt_1,149,99). coords(rwgt_1,180,153,99).  visible(rwgt_1,99). happensAt(walking(rwgt_1),99).
#show holdsAt/2.