
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

	orientation(fcgt_0,150,24). coords(fcgt_0,165,123,24).  visible(fcgt_0,24). happensAt(walking(fcgt_0),24). orientation(fcgt_4,166,24). coords(fcgt_4,103,214,24).  visible(fcgt_4,24). happensAt(walking(fcgt_4),24).
orientation(fcgt_0,150,25). coords(fcgt_0,166,123,25).  visible(fcgt_0,25). happensAt(walking(fcgt_0),25). orientation(fcgt_4,166,25). coords(fcgt_4,103,214,25).  visible(fcgt_4,25). happensAt(walking(fcgt_4),25).
orientation(fcgt_0,150,26). coords(fcgt_0,166,124,26).  visible(fcgt_0,26). happensAt(walking(fcgt_0),26). orientation(fcgt_4,166,26). coords(fcgt_4,102,214,26).  visible(fcgt_4,26). happensAt(walking(fcgt_4),26).
orientation(fcgt_0,150,27). coords(fcgt_0,166,124,27).  visible(fcgt_0,27). happensAt(walking(fcgt_0),27). orientation(fcgt_4,166,27). coords(fcgt_4,101,214,27).  visible(fcgt_4,27). happensAt(walking(fcgt_4),27).
orientation(fcgt_0,150,28). coords(fcgt_0,169,124,28).  visible(fcgt_0,28). happensAt(walking(fcgt_0),28). orientation(fcgt_4,166,28). coords(fcgt_4,101,212,28).  visible(fcgt_4,28). happensAt(walking(fcgt_4),28).
orientation(fcgt_0,150,29). coords(fcgt_0,171,125,29).  visible(fcgt_0,29). happensAt(walking(fcgt_0),29). orientation(fcgt_4,166,29). coords(fcgt_4,101,212,29).  visible(fcgt_4,29). happensAt(walking(fcgt_4),29).
orientation(fcgt_0,150,30). coords(fcgt_0,174,126,30).  visible(fcgt_0,30). happensAt(walking(fcgt_0),30). orientation(fcgt_4,166,30). coords(fcgt_4,101,212,30).  visible(fcgt_4,30). happensAt(walking(fcgt_4),30).
orientation(fcgt_0,150,31). coords(fcgt_0,175,127,31).  visible(fcgt_0,31). happensAt(walking(fcgt_0),31). orientation(fcgt_4,166,31). coords(fcgt_4,101,212,31).  visible(fcgt_4,31). happensAt(walking(fcgt_4),31).
orientation(fcgt_0,150,32). coords(fcgt_0,175,129,32).  visible(fcgt_0,32). happensAt(walking(fcgt_0),32). orientation(fcgt_4,166,32). coords(fcgt_4,101,212,32).  visible(fcgt_4,32). happensAt(walking(fcgt_4),32).
orientation(fcgt_0,150,33). coords(fcgt_0,177,130,33).  visible(fcgt_0,33). happensAt(walking(fcgt_0),33). orientation(fcgt_4,166,33). coords(fcgt_4,101,212,33).  visible(fcgt_4,33). happensAt(walking(fcgt_4),33).
orientation(fcgt_0,150,34). coords(fcgt_0,177,131,34).  visible(fcgt_0,34). happensAt(walking(fcgt_0),34). orientation(fcgt_4,166,34). coords(fcgt_4,101,212,34).  visible(fcgt_4,34). happensAt(walking(fcgt_4),34).
orientation(fcgt_0,150,35). coords(fcgt_0,177,132,35).  visible(fcgt_0,35). happensAt(walking(fcgt_0),35). orientation(fcgt_4,166,35). coords(fcgt_4,101,212,35).  visible(fcgt_4,35). happensAt(walking(fcgt_4),35).
orientation(fcgt_0,150,36). coords(fcgt_0,178,133,36).  visible(fcgt_0,36). happensAt(walking(fcgt_0),36). orientation(fcgt_4,169,36). coords(fcgt_4,101,212,36).  visible(fcgt_4,36). happensAt(walking(fcgt_4),36).
orientation(fcgt_0,150,37). coords(fcgt_0,178,133,37).  visible(fcgt_0,37). happensAt(walking(fcgt_0),37). orientation(fcgt_4,169,37). coords(fcgt_4,101,212,37).  visible(fcgt_4,37). happensAt(walking(fcgt_4),37).
orientation(fcgt_0,150,38). coords(fcgt_0,178,133,38).  visible(fcgt_0,38). happensAt(walking(fcgt_0),38). orientation(fcgt_4,169,38). coords(fcgt_4,101,212,38).  visible(fcgt_4,38). happensAt(walking(fcgt_4),38).
orientation(fcgt_0,150,39). coords(fcgt_0,178,133,39).  visible(fcgt_0,39). happensAt(walking(fcgt_0),39). orientation(fcgt_4,169,39). coords(fcgt_4,101,212,39).  visible(fcgt_4,39). happensAt(walking(fcgt_4),39).
orientation(fcgt_0,143,40). coords(fcgt_0,179,133,40).  visible(fcgt_0,40). happensAt(walking(fcgt_0),40). orientation(fcgt_4,169,40). coords(fcgt_4,101,212,40).  visible(fcgt_4,40). happensAt(walking(fcgt_4),40).
orientation(fcgt_0,143,41). coords(fcgt_0,180,135,41).  visible(fcgt_0,41). happensAt(walking(fcgt_0),41). orientation(fcgt_4,169,41). coords(fcgt_4,101,211,41).  visible(fcgt_4,41). happensAt(walking(fcgt_4),41).
orientation(fcgt_0,143,42). coords(fcgt_0,180,137,42).  visible(fcgt_0,42). happensAt(walking(fcgt_0),42). orientation(fcgt_4,169,42). coords(fcgt_4,101,210,42).  visible(fcgt_4,42). happensAt(walking(fcgt_4),42).
orientation(fcgt_0,143,43). coords(fcgt_0,181,139,43).  visible(fcgt_0,43). happensAt(walking(fcgt_0),43). orientation(fcgt_4,169,43). coords(fcgt_4,101,210,43).  visible(fcgt_4,43). happensAt(walking(fcgt_4),43).
orientation(fcgt_0,143,44). coords(fcgt_0,184,141,44).  visible(fcgt_0,44). happensAt(walking(fcgt_0),44). orientation(fcgt_4,169,44). coords(fcgt_4,100,210,44).  visible(fcgt_4,44). happensAt(walking(fcgt_4),44).
orientation(fcgt_0,143,45). coords(fcgt_0,186,143,45).  visible(fcgt_0,45). happensAt(walking(fcgt_0),45). orientation(fcgt_4,169,45). coords(fcgt_4,100,210,45).  visible(fcgt_4,45). happensAt(walking(fcgt_4),45).
orientation(fcgt_0,143,46). coords(fcgt_0,187,144,46).  visible(fcgt_0,46). happensAt(walking(fcgt_0),46). orientation(fcgt_4,169,46). coords(fcgt_4,100,210,46).  visible(fcgt_4,46). happensAt(walking(fcgt_4),46).
orientation(fcgt_0,143,47). coords(fcgt_0,189,145,47).  visible(fcgt_0,47). happensAt(walking(fcgt_0),47). orientation(fcgt_4,169,47). coords(fcgt_4,100,210,47).  visible(fcgt_4,47). happensAt(walking(fcgt_4),47).
orientation(fcgt_0,143,48). coords(fcgt_0,190,145,48).  visible(fcgt_0,48). happensAt(walking(fcgt_0),48). orientation(fcgt_4,166,48). coords(fcgt_4,100,210,48).  visible(fcgt_4,48). happensAt(walking(fcgt_4),48).
#show holdsAt/2.