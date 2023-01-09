
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

	orientation(lb2gt_0,141,12). coords(lb2gt_0,190,125,12).  visible(lb2gt_0,12). happensAt(inactive(lb2gt_0),12). orientation(lb2gt_1,130,12). coords(lb2gt_1,214,60,12).  visible(lb2gt_1,12). happensAt(walking(lb2gt_1),12).
orientation(lb2gt_0,141,13). coords(lb2gt_0,190,125,13).  visible(lb2gt_0,13). happensAt(inactive(lb2gt_0),13). orientation(lb2gt_1,130,13). coords(lb2gt_1,214,58,13).  visible(lb2gt_1,13). happensAt(walking(lb2gt_1),13).
orientation(lb2gt_0,141,14). coords(lb2gt_0,190,125,14).  visible(lb2gt_0,14). happensAt(inactive(lb2gt_0),14). orientation(lb2gt_1,130,14). coords(lb2gt_1,215,55,14).  visible(lb2gt_1,14). happensAt(walking(lb2gt_1),14).
orientation(lb2gt_0,141,15). coords(lb2gt_0,190,125,15).  visible(lb2gt_0,15). happensAt(inactive(lb2gt_0),15). orientation(lb2gt_1,130,15). coords(lb2gt_1,217,53,15).  visible(lb2gt_1,15). happensAt(walking(lb2gt_1),15).
orientation(lb2gt_0,141,16). coords(lb2gt_0,190,125,16).  visible(lb2gt_0,16). happensAt(inactive(lb2gt_0),16). orientation(lb2gt_1,130,16). coords(lb2gt_1,219,53,16).  visible(lb2gt_1,16). happensAt(walking(lb2gt_1),16).
orientation(lb2gt_0,141,17). coords(lb2gt_0,190,125,17).  visible(lb2gt_0,17). happensAt(inactive(lb2gt_0),17). orientation(lb2gt_1,130,17). coords(lb2gt_1,221,52,17).  visible(lb2gt_1,17). happensAt(walking(lb2gt_1),17).
orientation(lb2gt_0,141,18). coords(lb2gt_0,190,125,18).  visible(lb2gt_0,18). happensAt(inactive(lb2gt_0),18). orientation(lb2gt_1,130,18). coords(lb2gt_1,224,51,18).  visible(lb2gt_1,18). happensAt(walking(lb2gt_1),18).
orientation(lb2gt_0,141,19). coords(lb2gt_0,190,125,19).  visible(lb2gt_0,19). happensAt(inactive(lb2gt_0),19). orientation(lb2gt_1,130,19). coords(lb2gt_1,226,51,19).  visible(lb2gt_1,19). happensAt(walking(lb2gt_1),19).
orientation(lb2gt_0,141,20). coords(lb2gt_0,190,125,20).  visible(lb2gt_0,20). happensAt(inactive(lb2gt_0),20). orientation(lb2gt_1,130,20). coords(lb2gt_1,227,50,20).  visible(lb2gt_1,20). happensAt(walking(lb2gt_1),20).
orientation(lb2gt_0,141,21). coords(lb2gt_0,190,125,21).  visible(lb2gt_0,21). happensAt(inactive(lb2gt_0),21). orientation(lb2gt_1,130,21). coords(lb2gt_1,228,49,21).  visible(lb2gt_1,21). happensAt(walking(lb2gt_1),21).
orientation(lb2gt_0,141,22). coords(lb2gt_0,190,125,22).  visible(lb2gt_0,22). happensAt(inactive(lb2gt_0),22). orientation(lb2gt_1,124,22). coords(lb2gt_1,229,47,22).  visible(lb2gt_1,22). happensAt(walking(lb2gt_1),22).
orientation(lb2gt_0,141,23). coords(lb2gt_0,190,125,23).  visible(lb2gt_0,23). happensAt(inactive(lb2gt_0),23). orientation(lb2gt_1,121,23). coords(lb2gt_1,230,46,23).  visible(lb2gt_1,23). happensAt(walking(lb2gt_1),23).
orientation(lb2gt_0,141,24). coords(lb2gt_0,190,125,24).  visible(lb2gt_0,24). happensAt(inactive(lb2gt_0),24). orientation(lb2gt_1,117,24). coords(lb2gt_1,231,44,24).  visible(lb2gt_1,24). happensAt(walking(lb2gt_1),24).
orientation(lb2gt_0,141,25). coords(lb2gt_0,190,125,25).  visible(lb2gt_0,25). happensAt(active(lb2gt_0),25). orientation(lb2gt_1,112,25). coords(lb2gt_1,232,42,25).  visible(lb2gt_1,25). happensAt(walking(lb2gt_1),25).
orientation(lb2gt_0,141,26). coords(lb2gt_0,190,125,26).  visible(lb2gt_0,26). happensAt(active(lb2gt_0),26). orientation(lb2gt_1,119,26). coords(lb2gt_1,232,39,26).  visible(lb2gt_1,26). happensAt(walking(lb2gt_1),26).
orientation(lb2gt_0,141,27). coords(lb2gt_0,190,125,27).  visible(lb2gt_0,27). happensAt(active(lb2gt_0),27). orientation(lb2gt_1,117,27). coords(lb2gt_1,233,36,27).  visible(lb2gt_1,27). happensAt(walking(lb2gt_1),27).
orientation(lb2gt_0,141,28). coords(lb2gt_0,190,125,28).  visible(lb2gt_0,28). happensAt(active(lb2gt_0),28). orientation(lb2gt_1,119,28). coords(lb2gt_1,233,34,28).  visible(lb2gt_1,28). happensAt(walking(lb2gt_1),28).
orientation(lb2gt_0,141,29). coords(lb2gt_0,190,125,29).  visible(lb2gt_0,29). happensAt(active(lb2gt_0),29). orientation(lb2gt_1,115,29). coords(lb2gt_1,233,33,29).  visible(lb2gt_1,29). happensAt(walking(lb2gt_1),29).
orientation(lb2gt_0,141,30). coords(lb2gt_0,190,125,30).  visible(lb2gt_0,30). happensAt(active(lb2gt_0),30). orientation(lb2gt_1,119,30). coords(lb2gt_1,234,33,30).  visible(lb2gt_1,30). happensAt(walking(lb2gt_1),30).
orientation(lb2gt_0,141,31). coords(lb2gt_0,190,125,31).  visible(lb2gt_0,31). happensAt(active(lb2gt_0),31). orientation(lb2gt_1,118,31). coords(lb2gt_1,235,32,31).  visible(lb2gt_1,31). happensAt(walking(lb2gt_1),31).
orientation(lb2gt_0,141,32). coords(lb2gt_0,190,125,32).  visible(lb2gt_0,32). happensAt(active(lb2gt_0),32). orientation(lb2gt_1,116,32). coords(lb2gt_1,236,32,32).  visible(lb2gt_1,32). happensAt(walking(lb2gt_1),32).
orientation(lb2gt_0,141,33). coords(lb2gt_0,190,125,33).  visible(lb2gt_0,33). happensAt(active(lb2gt_0),33). orientation(lb2gt_1,116,33). coords(lb2gt_1,237,31,33).  visible(lb2gt_1,33). happensAt(walking(lb2gt_1),33).
orientation(lb2gt_0,141,34). coords(lb2gt_0,191,125,34).  visible(lb2gt_0,34). happensAt(active(lb2gt_0),34). orientation(lb2gt_1,113,34). coords(lb2gt_1,238,30,34).  visible(lb2gt_1,34). happensAt(walking(lb2gt_1),34).
orientation(lb2gt_0,141,35). coords(lb2gt_0,191,125,35).  visible(lb2gt_0,35). happensAt(active(lb2gt_0),35). orientation(lb2gt_1,110,35). coords(lb2gt_1,239,29,35).  visible(lb2gt_1,35). happensAt(walking(lb2gt_1),35).
orientation(lb2gt_0,141,36). coords(lb2gt_0,191,125,36).  visible(lb2gt_0,36). happensAt(active(lb2gt_0),36). orientation(lb2gt_1,109,36). coords(lb2gt_1,240,28,36).  visible(lb2gt_1,36). happensAt(walking(lb2gt_1),36).
#show holdsAt/2.